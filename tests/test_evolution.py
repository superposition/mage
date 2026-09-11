"""Contracts for the kernel evolution loop: parameter constraints, rendering, decisions.

The generated source is the artifact under test here, not an implementation detail:
it is what the loop edits, what the compiler consumes, and what the ledger records a
hash of. The GPU test at the end is skipped unless the cuda-oxide toolchain is present.
"""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BINARY = REPO / "examples" / "oxide" / "target" / "release" / "mage-oxide"


def load(name, base=SCRIPTS):
    spec = importlib.util.spec_from_file_location(name, base / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = load("evolve_render")
policy = load("evolve_policy")


# --- parameter constraints ---------------------------------------------------


def knobs(role):
    return render.MATMUL_KNOBS if role == "matmul" else render.LAYERNORM_KNOBS


def defaults(role):
    return dict(render.DEFAULT_MATMUL if role == "matmul" else render.DEFAULT_LAYERNORM)


def every_value(role):
    """One configuration per knob value, holding the rest at the incumbent."""
    for knob, values in knobs(role).items():
        for value in values:
            params = defaults(role)
            params[knob] = value
            yield knob, value, params


@pytest.mark.parametrize("role", ["matmul", "layernorm"])
def test_every_knob_value_either_validates_or_is_rejected_with_a_reason(role):
    refused = 0
    for knob, value, params in every_value(role):
        try:
            normalised = render.validate(role, params)
        except render.ParameterError as error:
            refused += 1
            assert str(error), f"{knob}={value} rejected without a reason"
            continue
        assert set(normalised) == set(knobs(role)), f"{knob}={value} normalised to {normalised}"
    # Both spaces must be mostly reachable, or the loop is searching nothing.
    assert refused < sum(len(values) for values in knobs(role).values())


def test_unsatisfiable_geometries_are_rejected_statically():
    cases = [
        # (role, override, expected fragment)
        ("matmul", {"block": (64, 64), "k_step": 128, "staging": "guarded"}, "shared memory"),
        ("matmul", {"block": (64, 128), "k_step": 64, "staging": "exact"}, "passes"),
        ("matmul", {"block": (32, 64), "k_step": 64, "staging": "shared"}, "shared"),
        ("matmul", {"block": (64, 65)}, "multiple of 4"),
        ("matmul", {"thread_tile": (8, 8), "block": (32, 32)}, "threads per block"),
        ("layernorm", {"warps_per_row": 8, "threads": 128}, "multiple of"),
    ]
    for role, override, fragment in cases:
        params = defaults(role)
        params.update(override)
        with pytest.raises(render.ParameterError) as raised:
            render.validate(role, params)
        assert fragment in str(raised.value), (override, str(raised.value))


def test_shape_that_the_tile_cannot_cover_is_rejected_before_a_build():
    params = dict(render.DEFAULT_MATMUL, block=(64, 64), k_step=32, staging="guarded")
    with pytest.raises(render.ParameterError) as raised:
        render.validate("matmul", params, (1024, 1020, 1024))
    assert "not a multiple" in str(raised.value)
    render.validate("matmul", params, (1024, 1024, 1024))


def test_layernorm_needs_rows_divisible_and_a_quad_width():
    params = dict(render.DEFAULT_LAYERNORM, warps_per_row=2, threads=256)
    render.validate("layernorm", params, (64, 768))
    with pytest.raises(render.ParameterError):
        render.validate("layernorm", params, (62, 768))
    with pytest.raises(render.ParameterError):
        render.validate("layernorm", params, (64, 767))


# --- rendering ---------------------------------------------------------------


def test_rendering_is_deterministic(tmp_path):
    first = tmp_path / "a.rs"
    second = tmp_path / "b.rs"
    args = (render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM,
            render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM)
    digest_a = render.write_candidates(*args, generation=1, dest=first)
    digest_b = render.write_candidates(*args, generation=1, dest=second)
    assert digest_a == digest_b
    assert first.read_bytes() == second.read_bytes()


def test_rendered_arrays_are_large_enough_for_their_tile(tmp_path):
    """The transposed A tile is [k][m]; the row-major one is [m][k]."""
    for transpose_a, expected_rows in ((True, 68), (False, 64)):
        params = dict(render.DEFAULT_MATMUL, block=(64, 64), k_step=32,
                      transpose_a=transpose_a, staging="guarded")
        render.write_candidates(params, render.DEFAULT_LAYERNORM,
                                params, render.DEFAULT_LAYERNORM,
                                generation=0, dest=tmp_path / "c.rs")
        source = (tmp_path / "c.rs").read_text()
        arrays = [int(size) for size in re.findall(r"SharedArray<f32, (\d+), 16>", source)]
        a_len, b_len = arrays[0], arrays[1]
        assert a_len == expected_rows * 32, (transpose_a, a_len)
        assert b_len == 64 * 32


def test_both_roles_are_rendered_with_their_own_metadata(tmp_path):
    best = dict(render.DEFAULT_MATMUL)
    new = dict(render.DEFAULT_MATMUL, k_step=32, staging="guarded", transpose_a=False)
    digest = render.write_candidates(best, render.DEFAULT_LAYERNORM,
                                     new, render.DEFAULT_LAYERNORM,
                                     generation=7, dest=tmp_path / "c.rs")
    source = (tmp_path / "c.rs").read_text()
    assert len(digest) == 64
    assert "pub const MATMUL_TILE_BEST: (u32, u32, u32) = (64, 64, 64);" in source
    assert "pub const MATMUL_TILE_NEW: (u32, u32, u32) = (64, 64, 32);" in source
    # The non-target operation keeps the incumbent parameters in both slots.
    assert source.count(render.params_json(render.DEFAULT_LAYERNORM)) == 4
    assert source.count("pub fn matmul_best(") == 1
    assert source.count("pub fn matmul_new(") == 1
    assert source.count("pub fn layernorm_best(") == 1
    assert source.count("pub fn layernorm_new(") == 1
    assert "generation 7" in source


# --- decisions ---------------------------------------------------------------


def test_accept_requires_clearing_the_gain_threshold():
    base = dict(correctness_passed=True, committed_medians=[100.0, 100.2, 100.1],
                min_gain=0.01, control_tolerance=0.03)
    thin = policy.decide(**base, new_medians=[99.4], best_medians=[100.0])
    assert thin["decision"] == "reject" and "1.00%" in thin["reason"]
    clear = policy.decide(**base, new_medians=[98.0], best_medians=[100.0])
    assert clear["decision"] == "accept" and clear["ratio"] == pytest.approx(0.98)


def test_a_single_boosted_round_does_not_carry_a_generation():
    """One clock-boosted round on either arm must not decide the generation."""
    # The candidate looks 20% faster in one round and identical in the others: the
    # old fastest-round rule accepted this at 0.80.
    boosted_candidate = policy.decide(
        correctness_passed=True, new_medians=[80.0, 100.0, 100.0],
        best_medians=[100.0, 100.0, 100.0], committed_medians=[100.0, 100.1, 100.0],
        min_gain=0.02, control_tolerance=0.03)
    assert boosted_candidate["decision"] == "reject"
    assert boosted_candidate["ratio"] == pytest.approx(1.0)
    # The incumbent looks 20% faster in one round: no free pass either.
    boosted_incumbent = policy.decide(
        correctness_passed=True, new_medians=[100.0, 100.0, 100.0],
        best_medians=[80.0, 100.0, 100.0], committed_medians=[100.0, 100.1, 100.0],
        min_gain=0.02, control_tolerance=0.03)
    assert boosted_incumbent["decision"] == "reject"
    assert boosted_incumbent["ratio"] == pytest.approx(1.0)


def test_a_consistent_win_is_accepted():
    verdict = policy.decide(
        correctness_passed=True, new_medians=[98.0, 98.2, 98.1],
        best_medians=[100.0, 100.0, 100.1], committed_medians=[100.0, 100.2, 100.1],
        min_gain=0.01, control_tolerance=0.03)
    assert verdict["decision"] == "accept"
    assert verdict["ratio"] == pytest.approx(0.98, abs=0.001)
    assert len(verdict["ratios"]) == 3


def test_a_failed_correctness_gate_blocks_an_otherwise_faster_candidate():
    verdict = policy.decide(correctness_passed=False, new_medians=[50.0],
                            best_medians=[100.0], committed_medians=[100.0, 100.0],
                            min_gain=0.01, control_tolerance=0.03)
    assert verdict["decision"] == "reject" and "correctness" in verdict["reason"]


def test_a_drifting_control_makes_the_round_unjudgeable():
    verdict = policy.decide(correctness_passed=True, new_medians=[50.0],
                            best_medians=[100.0],
                            committed_medians=[100.0, 100.0, 113.0],
                            min_gain=0.01, control_tolerance=0.03)
    assert verdict["decision"] == "reject" and verdict["noisy"] is True
    assert verdict["control_drift"] > 1.03


def test_proposals_change_one_knob_and_visit_distinct_tuples():
    for role in ("matmul", "layernorm"):
        proposer = policy.Proposer(knobs(role), defaults(role), seed=3)
        seen = set()
        for _ in range(12):
            proposal = proposer.propose()
            if proposal is None:
                break
            changed = [name for name in knobs(role)
                       if proposal.params[name] != proposer.best[name]]
            assert changed == [proposal.knob], (role, changed)
            key = policy.canonical(proposal.params)
            assert key not in seen
            seen.add(key)
            proposer.record(proposal, accepted=False)


def test_a_value_rejected_twice_is_not_proposed_again():
    role = "layernorm"
    proposer = policy.Proposer(knobs(role), defaults(role), seed=1, freeze_after=2)
    counts = {}
    for _ in range(8):
        proposal = proposer.propose()
        if proposal is None:
            break
        key = (proposal.knob, policy.canonical({"value": proposal.after}))
        counts[key] = counts.get(key, 0) + 1
        proposer.record(proposal, accepted=False)
    assert max(counts.values()) <= 2, counts


# --- the generated surface on a GPU (skipped without the toolchain) ----------


@pytest.mark.skipif(not BINARY.is_file() or shutil.which("cargo") is None,
                    reason="cuda-oxide build or binary not available")
def test_generated_variant_matches_the_reference(tmp_path):
    experiment = load("experiment", REPO / "examples" / "oxide")
    render.write_candidates(render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM,
                            render.DEFAULT_MATMUL, render.DEFAULT_LAYERNORM,
                            generation=0)
    build = subprocess.run(
        ["bash", "-lc",
         "source scripts/oxide-env.sh && cd examples/oxide && "
         "CARGO_BUILD_JOBS=2 cargo oxide build --arch sm_89"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    directory = tmp_path / "matmul"
    experiment.generate(directory, "matmul", [256, 256, 128], warmup=2, iterations=5)
    manifest = json.loads((directory / "input.json").read_text())
    manifest["variant"] = "best"
    (directory / "input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run = subprocess.run([str(BINARY), str(directory)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    result = experiment.validate(directory)
    assert result["passed"] and result["max_abs_error"] <= 1e-4
    timing = json.loads((directory / "rust-timing.json").read_text())
    assert timing["variant"] == "best" and timing["variant_kernel"] == "matmul_best"
