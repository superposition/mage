"""Render the generated candidate kernels for the evolution loop.

`candidates.rs` is produced from two templates that live beside the committed
kernels:

- `examples/oxide/src/candidates.rs.tmpl` -- the module, its launch metadata, and
  the two rendered kernel bodies.
- `examples/oxide/src/candidates_body.rs.tmpl` -- one parameterised kernel pair
  (matmul + layernorm) with `//>> if NAME / else / end` control lines and
  `{{PLACEHOLDER}}` substitution points.

Two roles are rendered per build, `best` (the loop's incumbent) and `new` (the
proposal), so one build measures both under identical conditions. A manifest with
no `variant` key keeps running the committed kernels in `main.rs`.

Knob values are geometry, not free text: `validate` rejects combinations the
generated source cannot honour (thread counts, quad alignment, the 48 KB static
shared-memory limit) before anything is compiled.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = REPO_ROOT / "examples" / "oxide" / "src" / "candidates.rs"
WRAPPER_TEMPLATE = REPO_ROOT / "examples" / "oxide" / "src" / "candidates.rs.tmpl"
BODY_TEMPLATE = REPO_ROOT / "examples" / "oxide" / "src" / "candidates_body.rs.tmpl"
READ_ONLY_TEMPLATES = (
    "examples/oxide/src/candidates.rs.tmpl",
    "examples/oxide/src/candidates_body.rs.tmpl",
)

# Static shared memory a block may declare without an opt-in carveout on sm_89.
SHARED_BUDGET_BYTES = 46_080
MAX_THREADS = 1024
WARP = 32
STAGING_MODES = ("shared", "exact", "guarded")

# Ordered by prior knowledge: the tile shape and K step dominated the measured
# history, the staging layout switches were worth 1.6x, the register tile hurt
# past 4x4. The loop re-ranks them from its own ledger.
MATMUL_KNOBS: dict[str, list] = {
    # Prior order follows the measured history in mage-003: the two staging-layout
    # changes were worth more than any tile-shape change, and the K step followed
    # them. The loop re-ranks this from its own ledger as soon as it has results.
    "transpose_a": [False, True],
    "quad_stage": [False, True],
    "k_step": [32, 64, 16, 128],
    "block": [(64, 64), (128, 64), (64, 128), (32, 64), (32, 32), (128, 32), (128, 128)],
    # Guard-free staging reuses one quad decomposition for A and B and needs no
    # range test, which is only sound when both tiles consume identical quad
    # indices. `exact` keeps the guard-free form for any tile whose two quad counts
    # fill the same whole number of passes; `guarded` fits every geometry but pays
    # a branch per target.
    "staging": ["shared", "exact", "guarded"],
    "thread_tile": [(4, 4), (8, 4), (4, 8), (8, 8)],
}

LAYERNORM_KNOBS: dict[str, list] = {
    "warps_per_row": [1, 2, 4, 8],
    "threads": [256, 128, 512],
}

# The incumbents reproduce the committed kernels: matmul's register-tiled 64x64
# fast path, and layer_norm_pair's two warps per row in a 256-thread block.
DEFAULT_MATMUL: dict[str, object] = {
    "block": (64, 64),
    "k_step": 64,
    "transpose_a": True,
    "quad_stage": True,
    "staging": "shared",
    "thread_tile": (4, 4),
}
DEFAULT_LAYERNORM: dict[str, object] = {"warps_per_row": 2, "threads": 256}


class ParameterError(ValueError):
    """A candidate the generated source cannot honour."""


def _as_pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
        if pair[0] > 0 and pair[1] > 0:
            return pair
    raise ParameterError(f"{name} must be a pair of positive integers, got {value!r}")


def _as_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParameterError(f"{name} must be a positive integer, got {value!r}")
    return value


def matmul_geometry(params: dict[str, object]) -> dict[str, object]:
    """Derive every number the matmul body template needs."""
    block_m, block_n = _as_pair(params["block"], "block")
    k_step = _as_positive_int(params["k_step"], "k_step")
    tile_m, tile_n = _as_pair(params["thread_tile"], "thread_tile")
    transpose_a = bool(params["transpose_a"])
    quad_stage = bool(params["quad_stage"])
    staging = str(params["staging"])
    if staging not in STAGING_MODES:
        raise ParameterError(f"staging must be one of {', '.join(STAGING_MODES)}; got {staging!r}")

    for name, value in (("block_m", block_m), ("block_n", block_n), ("tile_m", tile_m), ("tile_n", tile_n)):
        if value % 4 != 0:
            raise ParameterError(f"{name} ({value}) must be a multiple of 4 for 128-bit shared access")
    if k_step % 4 != 0:
        raise ParameterError(f"k_step ({k_step}) must be a multiple of 4 for 128-bit staging loads")

    threads_x = block_n // tile_n
    threads_y = block_m // tile_m
    threads = threads_x * threads_y
    if threads > MAX_THREADS:
        raise ParameterError(f"{threads} threads per block exceeds {MAX_THREADS}")
    if threads % WARP != 0:
        raise ParameterError(f"{threads} threads per block is not a whole number of warps")

    a_quads = block_m * k_step // 4
    b_quads = k_step * block_n // 4
    a_passes, a_remainder = divmod(a_quads, threads)
    b_passes, b_remainder = divmod(b_quads, threads)
    if staging == "shared":
        if not (block_m == block_n == k_step):
            raise ParameterError(
                "staging=shared reuses one quad decomposition for A and B, so it needs "
                f"block_m == block_n == k_step; got {block_m}x{block_n} with k step {k_step}"
            )
        passes, shared_rowcol, guarded = a_passes, True, False
    elif staging == "exact":
        if a_remainder or b_remainder or a_passes != b_passes:
            raise ParameterError(
                f"staging=exact needs both tiles to fill the same whole number of passes; "
                f"got {a_quads} A and {b_quads} B quads over {threads} threads"
            )
        passes, shared_rowcol, guarded = a_passes, False, False
    else:
        passes = max(a_quads, b_quads + threads - 1) // threads
        shared_rowcol, guarded = False, True
    # Transposed staging keeps A as [k][m] with a padded row stride; the row-major
    # layout is [m][k] and therefore needs block_m rows, not k_step of them.
    at_stride = block_m + 4 if transpose_a else k_step
    at_len = at_stride * k_step if transpose_a else block_m * k_step
    bs_len = block_n * k_step
    shared_bytes = (at_len + bs_len) * 4
    if shared_bytes > SHARED_BUDGET_BYTES:
        raise ParameterError(
            f"{shared_bytes} B of static shared memory exceeds the {SHARED_BUDGET_BYTES} B budget"
        )

    return {
        "BM": block_m,
        "BN": block_n,
        "BK": k_step,
        "TM": tile_m,
        "TN": tile_n,
        "TX": threads_x,
        "THREADS": threads,
        "PASSES": passes,
        "A_QUADS": a_quads,
        "B_QUADS": b_quads,
        "A_QUADS_PER_ROW": k_step // 4,
        "B_QUADS_PER_ROW": block_n // 4,
        "AT_STRIDE": at_stride,
        "AT_LEN": at_len,
        "BS_LEN": bs_len,
        "TM_QUADS": tile_m // 4,
        "TN_QUADS": tile_n // 4,
        "AT_QUAD_STRIDE": at_stride // 4,
        "TRANSPOSE_A": transpose_a,
        "QUAD_STAGE": quad_stage,
        "GUARDED_A": guarded,
        "GUARDED_B": guarded,
        "SHARED_ROWCOL": shared_rowcol,
        "A_ROW": "row" if shared_rowcol else "q / %d" % (k_step // 4),
        "A_COL": "col" if shared_rowcol else "(q %% %d) * 4" % (k_step // 4),
        "B_ROW": "row" if shared_rowcol else "q / %d" % (block_n // 4),
        "B_COL": "col" if shared_rowcol else "(q %% %d) * 4" % (block_n // 4),
    }


def layernorm_geometry(params: dict[str, object]) -> dict[str, object]:
    """Derive every number the layernorm body template needs."""
    warps_per_row = _as_positive_int(params["warps_per_row"], "warps_per_row")
    threads = _as_positive_int(params["threads"], "threads")
    if threads > MAX_THREADS or threads % WARP != 0:
        raise ParameterError(f"threads ({threads}) must be a whole number of warps up to {MAX_THREADS}")
    warps = threads // WARP
    if warps % warps_per_row != 0:
        raise ParameterError(f"{warps} warps per block is not a multiple of {warps_per_row} warps per row")
    return {
        "LN_WPR": warps_per_row,
        "LN_ROWS": warps // warps_per_row,
        # Namespaced so a body render can hold both kernels' geometry at once.
        "LN_THREADS": threads,
        "PARTIAL_LEN": 2 * warps,
    }


def geometry(role: str, params: dict[str, object]) -> dict[str, object]:
    if role == "matmul":
        return matmul_geometry(params)
    if role == "layernorm":
        return layernorm_geometry(params)
    raise ParameterError(f"unknown role {role!r}")


def validate(role: str, params: dict[str, object], shape: tuple[int, ...] | None = None) -> dict[str, object]:
    """Normalise `params` for `role`, rejecting what the template cannot honour.

    When `shape` is given, the tile divisibility the host enforces is checked here
    as well so an unsatisfiable candidate costs a function call, not a build.
    """
    knobs = MATMUL_KNOBS if role == "matmul" else LAYERNORM_KNOBS
    if role not in ("matmul", "layernorm"):
        raise ParameterError(f"unknown role {role!r}")
    unknown = sorted(set(params) - set(knobs))
    if unknown:
        raise ParameterError(f"unknown {role} knob(s): {', '.join(unknown)}")
    missing = sorted(set(knobs) - set(params))
    if missing:
        raise ParameterError(f"missing {role} knob(s): {', '.join(missing)}")

    if role == "matmul":
        normalised: dict[str, object] = {
            "block": _as_pair(params["block"], "block"),
            "k_step": _as_positive_int(params["k_step"], "k_step"),
            "transpose_a": bool(params["transpose_a"]),
            "quad_stage": bool(params["quad_stage"]),
            "staging": str(params["staging"]),
            "thread_tile": _as_pair(params["thread_tile"], "thread_tile"),
        }
    else:
        normalised = {
            "warps_per_row": _as_positive_int(params["warps_per_row"], "warps_per_row"),
            "threads": _as_positive_int(params["threads"], "threads"),
        }

    derived = geometry(role, normalised)
    if shape is not None:
        if role == "matmul":
            if len(shape) != 3:
                raise ParameterError(f"matmul shape must be m,n,k, got {shape!r}")
            m, n, k = (int(v) for v in shape)
            block_m, block_n, k_step = derived["BM"], derived["BN"], derived["BK"]
            if m % block_m or n % block_n or k % k_step:
                raise ParameterError(
                    f"shape [{m}, {n}, {k}] is not a multiple of the {block_m}x{block_n} tile with k step {k_step}"
                )
        else:
            if len(shape) != 2:
                raise ParameterError(f"layernorm shape must be rows,width, got {shape!r}")
            rows, width = (int(v) for v in shape)
            if rows % derived["LN_ROWS"]:
                raise ParameterError(f"rows ({rows}) is not a multiple of {derived['LN_ROWS']} rows per block")
            if width % 4:
                raise ParameterError(f"width ({width}) must be a multiple of 4 for the quad path")
    return normalised


def params_json(params: dict[str, object]) -> str:
    """Canonical JSON for provenance and hashing."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_text(path.read_text(encoding="utf-8"))


def _condition(expression: str, context: dict[str, object]) -> bool:
    """Evaluate a `//>> if` condition: `FLAG` or `FLAG > NUMBER`."""
    match = re.fullmatch(r"\s*(\w+)\s*(?:(>=|<=|==|>|<)\s*(\d+))?\s*", expression)
    if not match:
        raise ParameterError(f"unparsable control condition {expression!r}")
    name, operator, literal = match.groups()
    if name not in context:
        raise ParameterError(f"unknown control flag {name!r}")
    if operator is None:
        return bool(context[name])
    left, right = int(context[name]), int(literal)
    return {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right, "==": left == right}[operator]


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render(text: str, context: dict[str, object], source: str) -> str:
    """Resolve `//>>` control lines, then substitute `{{PLACEHOLDER}}` tokens."""
    out: list[str] = []
    stack: list[tuple[bool, bool]] = []  # (branch active, branch taken)
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.strip()
        if stripped.startswith("//>>"):
            directive = stripped[4:].strip()
            if directive.startswith("if "):
                taken = _condition(directive[3:], context)
                stack.append((taken, taken))
            elif directive == "else":
                if not stack:
                    raise ParameterError(f"{source}:{number}: else without if")
                active, taken = stack.pop()
                stack.append((not active and not taken, True))
            elif directive == "end":
                if not stack:
                    raise ParameterError(f"{source}:{number}: end without if")
                stack.pop()
            else:
                raise ParameterError(f"{source}:{number}: unknown directive {directive!r}")
            continue
        if all(active for active, _ in stack):
            out.append(line)
    if stack:
        raise ParameterError(f"{source}: unterminated //>> if block")
    rendered = "".join(out)

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            raise ParameterError(f"{source}: no value for placeholder {{{{{name}}}}}")
        return _format(context[name])

    return re.sub(r"\{\{(\w+)\}\}", substitute, rendered)


def render_body(
    matmul: dict[str, object],
    layernorm: dict[str, object],
    suffix: str,
    role_label: str,
) -> str:
    """Render both kernels for one role (`best` or `new`)."""
    context: dict[str, object] = {
        "SUFFIX": suffix,
        "ROLE": role_label,
        "PARAMS_JSON": params_json(matmul),
        "MATMUL_PARAMS_JSON": params_json(matmul),
        "LAYERNORM_PARAMS_JSON": params_json(layernorm),
    }
    context.update(matmul_geometry(matmul))
    context.update(layernorm_geometry(layernorm))
    return _render(BODY_TEMPLATE.read_text(encoding="utf-8"), context, BODY_TEMPLATE.name)


def render_module(
    best_matmul: dict[str, object],
    best_layernorm: dict[str, object],
    new_matmul: dict[str, object],
    new_layernorm: dict[str, object],
    *,
    generation: int,
) -> str:
    """Render the whole `candidates.rs` source."""
    best_mm = matmul_geometry(best_matmul)
    best_ln = layernorm_geometry(best_layernorm)
    new_mm = matmul_geometry(new_matmul)
    new_ln = layernorm_geometry(new_layernorm)
    bodies = [
        render_body(best_matmul, best_layernorm, "best", "incumbent"),
        "",
        render_body(new_matmul, new_layernorm, "new", "candidate"),
    ]
    context: dict[str, object] = {
        "BODIES": "\n".join(bodies),
        "GENERATION": generation,
        "RENDERED_AT": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "WRAPPER_SHA256": _hash_file(WRAPPER_TEMPLATE),
        "BODY_SHA256": _hash_file(BODY_TEMPLATE),
        "BEST_BM": best_mm["BM"],
        "BEST_BN": best_mm["BN"],
        "BEST_BK": best_mm["BK"],
        "BEST_BX": best_mm["TX"],
        "BEST_BY": best_mm["THREADS"] // best_mm["TX"],
        "BEST_LN_ROWS": best_ln["LN_ROWS"],
        "BEST_THREADS": best_ln["LN_THREADS"],
        "BEST_MATMUL_PARAMS_JSON": params_json(best_matmul),
        "BEST_LAYERNORM_PARAMS_JSON": params_json(best_layernorm),
        "NEW_BM": new_mm["BM"],
        "NEW_BN": new_mm["BN"],
        "NEW_BK": new_mm["BK"],
        "NEW_BX": new_mm["TX"],
        "NEW_BY": new_mm["THREADS"] // new_mm["TX"],
        "NEW_LN_ROWS": new_ln["LN_ROWS"],
        "NEW_THREADS": new_ln["LN_THREADS"],
        "NEW_MATMUL_PARAMS_JSON": params_json(new_matmul),
        "NEW_LAYERNORM_PARAMS_JSON": params_json(new_layernorm),
    }
    return _render(WRAPPER_TEMPLATE.read_text(encoding="utf-8"), context, WRAPPER_TEMPLATE.name)


def write_candidates(
    best_matmul: dict[str, object],
    best_layernorm: dict[str, object],
    new_matmul: dict[str, object],
    new_layernorm: dict[str, object],
    *,
    generation: int,
    dest: Path | None = None,
) -> str:
    """Render `candidates.rs`, write it, and return the sha256 of what was written."""
    best_matmul = validate("matmul", best_matmul)
    best_layernorm = validate("layernorm", best_layernorm)
    new_matmul = validate("matmul", new_matmul)
    new_layernorm = validate("layernorm", new_layernorm)
    source = render_module(
        best_matmul, best_layernorm, new_matmul, new_layernorm, generation=generation
    )
    target = CANDIDATES_PATH if dest is None else Path(dest)
    target.write_text(source, encoding="utf-8")
    return _hash_text(source)


if __name__ == "__main__":  # render the incumbents for both roles: a manual sanity check
    digest = write_candidates(
        DEFAULT_MATMUL, DEFAULT_LAYERNORM, DEFAULT_MATMUL, DEFAULT_LAYERNORM, generation=0
    )
    print(f"wrote {CANDIDATES_PATH.relative_to(REPO_ROOT)} sha256={digest}")
