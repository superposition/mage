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

The code and detailed evidence are maintained [in GitHub](https://github.com/superposition/mage). The [Superposition journal](https://superposition.github.io/) follows the motivation behind the work.
