"""Proposal and decision rules for the kernel evolution loop.

Everything here is pure: no filesystem, no subprocesses, no hardware, and no
import of `evolve_render`. Knob sets are passed in, so the rules can be
exercised with plain assertions on any machine.

The loop is coordinate descent over one op's knobs. `Proposer` holds the
incumbent parameters, the measured payoff of each knob's accepted changes, and
the history of what has already been rejected, and hands back one proposal at a
time. `decide` turns a generation's round medians plus its correctness gate into
accept/reject, with the committed-kernel control as the drift guard.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

# A knob is frozen for the current incumbent after this many rejections, so the
# loop stops paying build+measure cost for a direction the evidence already
# refused.
FREEZE_AFTER = 2


def canonical(params: dict) -> str:
    """Canonical JSON for a params dict: the identity used for deduplication."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def format_value(value: object) -> str:
    """Compact rendering of a knob value for `hypothesis` text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def control_drift(medians: list[float]) -> float:
    """`max/min` of the control (committed-kernel) round medians.

    The control runs in every round of every generation, so its spread is the
    session's own noise floor. A generation whose control spreads wider than the
    tolerance cannot judge a candidate and is rejected as noisy. With no control
    sample at all the drift is unknown (`inf`); with exactly one, no spread was
    observed, so the measured drift is 1.0x.
    """
    usable = [float(m) for m in medians if m is not None and m > 0]
    if not usable:
        return float("inf")
    if len(usable) == 1:
        return 1.0
    return max(usable) / min(usable)


def decide(
    *,
    correctness_passed: bool,
    new_medians: list[float],
    best_medians: list[float],
    committed_medians: list[float],
    min_gain: float,
    control_tolerance: float,
) -> dict:
    """Apply the accept rule and return the recorded decision fields.

    Accept iff the candidate passed the correctness gate, the committed control
    stayed within tolerance, and the **median of the paired per-round ratios**
    clears the gain threshold: `median(new[i] / best[i]) < 1 - min_gain`.

    Pairing matters. The machine produces clock-boost excursions of about 10% that
    land on either arm, so a rule built on the fastest round of each arm can be
    carried by a single boosted round on one side, and a rule built on arm medians
    can be carried by a boosted round anywhere in that arm. Pairing each round with
    its own partner makes an excursion perturb one ratio instead of one arm, and the
    median over rounds decides whether the shift is systematic. With a single round
    the median is that round's ratio, so `--rounds 1` still works.
    """
    result = {
        "decision": "reject",
        "noisy": False,
        "control_drift": None,
        "ratio": None,
        "ratios": [],
        "reason": "",
    }
    if not committed_medians:
        result["reason"] = "no committed-kernel control measurement in this generation"
        return result
    drift = control_drift(committed_medians)
    result["control_drift"] = None if drift == float("inf") else drift
    # `control_tolerance` is relative: a 3% budget rejects max/min above 1.03.
    result["noisy"] = drift > 1.0 + control_tolerance
    if not correctness_passed:
        result["reason"] = "correctness gate failed; timing is not accepted"
        return result
    if not new_medians or not best_medians:
        result["reason"] = "no usable timing on one arm"
        return result
    if result["noisy"]:
        result["reason"] = (
            f"control drifted {drift - 1.0:+.2%} ({drift:.4f}x) in this generation "
            f"(tolerance {control_tolerance:.2%}); re-measure before judging"
        )
        return result
    ratios = [float(new) / float(best) for new, best in zip(new_medians, best_medians)]
    ratio = statistics.median(ratios)
    result["ratio"] = ratio
    result["ratios"] = ratios
    if ratio < 1.0 - min_gain:
        result["decision"] = "accept"
        result["reason"] = (
            f"new is {1.0 - ratio:.2%} faster than the incumbent "
            f"(required >{min_gain:.2%})"
        )
    else:
        result["reason"] = (
            f"gain {1.0 - ratio:+.2%} does not clear the {min_gain:.2%} threshold"
        )
    return result


@dataclass
class Proposal:
    """One single-knob change to the incumbent."""

    params: dict
    knob: str
    before: object
    after: object
    hypothesis: str
    attempt: int
    prior_payoff: float


class Proposer:
    """Coordinate descent over one op's knobs, driven by measured payoff.

    Rules, in the order they bind:

    - one knob changes per proposal, so a generation's delta is attributable;
    - pending knobs are ordered by measured payoff (descending), then by the
      knob's prior order in `knobs` - the prior is the exploration order until
      the ledger has something to say;
    - a `(knob, value)` pair rejected under the current incumbent is never
      proposed again; a knob is frozen after `FREEZE_AFTER` rejections under
      that incumbent;
    - an exact params tuple is never regenerated;
    - a change that is accepted becomes the new incumbent, which clears every
      rejection recorded against the old one (but not the tried-tuple set).
    """

    def __init__(
        self,
        knobs: dict[str, list],
        default: dict,
        *,
        seed: int = 0,
        freeze_after: int = FREEZE_AFTER,
    ):
        self.knobs = {name: list(values) for name, values in knobs.items()}
        self.order = list(knobs)
        self.best = dict(default)
        self.seed = seed
        self.freeze_after = freeze_after
        self.payoff: dict[str, float] = {}
        self.acceptances: dict[str, int] = {}
        self.last_gain: dict[str, float] = {}
        self.attempts: dict[str, int] = {}
        self.incumbent = canonical(self.best)
        self.rejected: set[tuple[str, str, str]] = set()
        self.tried: set[str] = {self.incumbent}

    # -- inspection ---------------------------------------------------------

    def ordered_payoffs(self) -> list[dict]:
        """Knob payoffs, best first, ties broken by the prior order."""
        indexed = {name: index for index, name in enumerate(self.order)}
        ranked = sorted(
            self.order, key=lambda name: (-self.payoff.get(name, 0.0), indexed[name])
        )
        return [
            {
                "knob": name,
                "payoff": self.payoff.get(name, 0.0),
                "acceptances": self.acceptances.get(name, 0),
                "last_gain": self.last_gain.get(name),
            }
            for name in ranked
        ]

    def _tried(self, params: dict) -> bool:
        return canonical(params) in self.tried

    def _value_available(self, knob: str, value: object) -> bool:
        if value == self.best[knob]:
            return False
        signature = (
            self.incumbent,
            knob,
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        )
        if signature in self.rejected:
            return False
        candidate = dict(self.best)
        candidate[knob] = value
        return not self._tried(candidate)

    def _next_value(self, knob: str):
        for value in self.knobs[knob]:
            if self._value_available(knob, value):
                return value
        return None

    def exhausted(self) -> bool:
        return self.propose() is None

    # -- the loop's interface ----------------------------------------------

    def propose(self) -> Proposal | None:
        """The next single-knob change, or `None` when nothing is left to try."""
        pending = []
        for index, knob in enumerate(self.order):
            if self.attempts.get(knob, 0) >= self.freeze_after:
                continue
            value = self._next_value(knob)
            if value is None:
                continue
            pending.append((knob, value, index))
        if not pending:
            return None
        pending.sort(key=lambda item: (-self.payoff.get(item[0], 0.0), item[2]))
        knob, value, index = pending[0]
        before = self.best[knob]
        params = dict(self.best)
        params[knob] = value
        attempt = self.attempts.get(knob, 0) + 1
        prior_payoff = self.payoff.get(knob, 0.0)
        if attempt == 1:
            hypothesis = (
                f"knob={knob} {format_value(before)}->{format_value(value)}; "
                f"first attempt at this knob, prior order {index + 1}/{len(self.order)}"
            )
        else:
            hypothesis = (
                f"knob={knob} {format_value(before)}->{format_value(value)}; "
                f"attempt {attempt} at this knob, prior payoff {prior_payoff:+.1%}"
            )
        self.tried.add(canonical(params))
        return Proposal(
            params=params,
            knob=knob,
            before=before,
            after=value,
            hypothesis=hypothesis,
            attempt=attempt,
            prior_payoff=prior_payoff,
        )

    def record(self, proposal: Proposal, *, accepted: bool, gain: float = 0.0) -> None:
        """Fold a generation's outcome back into the search state."""
        if accepted:
            self.payoff[proposal.knob] = self.payoff.get(proposal.knob, 0.0) + gain
            self.acceptances[proposal.knob] = self.acceptances.get(proposal.knob, 0) + 1
            self.last_gain[proposal.knob] = gain
            self.best = dict(proposal.params)
            self.incumbent = canonical(self.best)
            self.attempts = {}
            return
        self.attempts[proposal.knob] = self.attempts.get(proposal.knob, 0) + 1
        value = json.dumps(proposal.after, sort_keys=True, separators=(",", ":"))
        self.rejected.add((self.incumbent, proposal.knob, value))
