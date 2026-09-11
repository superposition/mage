---
title: Field notes
permalink: /experiments/
description: Follow a mathematical idea into an experiment, then bring the evidence back to the idea.
---
### 001 — Computation is also movement

A matrix product contains opportunities for reuse. A reduction requires partial answers to meet. A neighborhood sum turns local relationships into a larger computation.

[Read the first investigation]({{ '/experiments/mage-001/' | relative_url }}).

### 002 — Keeping values close to the arithmetic

A reduction ends in one value per row, so partial answers have to meet somewhere. A tile decides how often shared memory is read. Two Rust kernels were rearranged around both questions.

[Read the second investigation]({{ '/experiments/mage-002/' | relative_url }}).

### 003 — What the kernels were short of

A ratio between two register shapes separates shared-memory instructions from arithmetic. The count of resident threads decides how much memory traffic a grid can keep in flight. Both signals chose the next change, and several attempts measured worse than what they replaced.

[Read the third investigation]({{ '/experiments/mage-003/' | relative_url }}).

### 004 — What the compiler chose

A tile compiler takes over the register tiles, the shared-memory layouts and the barriers. It wins one operation, matches another, and loses the two matmul-shaped ones — and most of the gap in the timing column turns out to be the launch path rather than the kernel. The library's own search also beat the hand-picked tile ladder.

[Read the fourth investigation]({{ '/experiments/mage-004/' | relative_url }}).

### 005 — What the loop found, and what answered it

A benchmark that proposes its own variants rediscovered two changes that had been chosen by hand, and kept a third that turned out to be half of a combination faster than either half. Two of its own claims were withdrawn by better measurement, and the hand-written pipeline that landed afterwards went further still.

[Read the fifth investigation]({{ '/experiments/mage-005/' | relative_url }}).

### 006 — The load in flight, and the shape Triton uses

A matrix multiply that waits for its own tile loads can be given a second buffer and an asynchronous copy, so the next tile arrives while the current one is multiplied. A layer norm can be rearranged into the shape the other kernel already uses, read from its generated code. The second rewrite narrows the gap and does not close it.

[Read the sixth investigation]({{ '/experiments/mage-006/' | relative_url }}).

The code and detailed evidence are maintained [in GitHub](https://github.com/superposition/mage). The [Superposition journal](https://superposition.github.io/) follows the motivation behind the work.
