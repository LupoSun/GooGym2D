# GooGym2D

GooGym2D is a Gymnasium-based 2D bridge-construction environment extracted from the research codebase behind two 2026 papers on learning structural design intuition from human demonstrations: **IASS 2026** (*Capturing Structural Intuition*) and **ACADIA 2026** (*From Teaching to Teaming*). This public repo intentionally contains only the core environment and structural-analysis logic needed for reinforcement learning and imitation learning research.

Project website: <https://luposun.github.io/GooGym2D/>

What is included:

- A headless `gymnasium.Env` for incremental bridge construction
- Graph-state utilities for replay, hindsight reconstruction, and inspection
- A lightweight axial truss FEM pipeline for terminal structural evaluation
- Focused tests and a small, documented Python package

The interactive GUIs, the browser-based classroom collector, the imitation-learning
training stack, and the assistive co-design mode described in the papers are not
part of this package.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import gymnasium as gym
import googym2d

env = gym.make(googym2d.ENV_ID)
obs, info = env.reset(seed=0, options={"chasm_width": 10.0})

action = np.array([0.0, 0.0, 0.0, 3.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

# The public env keeps graph editing and terminal evaluation separate.
obs, reward, terminated, truncated, info = env.unwrapped.finalize_episode()
```

If you prefer direct construction:

```python
from googym2d import BridgeBuildEnv

env = BridgeBuildEnv(endpoint_mode="training_zones")
obs, info = env.reset(seed=0)
```

The environment is headless by design: passing a `render_mode` raises `ValueError`,
and `render()` is a no-op.

## Environment Design

Each action attempts to place one bar in anchor-first form:

```text
[anchor_x, anchor_y, theta, length]
```

- `anchor_x, anchor_y`: where the member starts
- `theta`: direction in radians
- `length`: member length, clipped to the environment maximum

Observation is a flat vector of shape `1 + 5 * MAX_BARS`, which is **126** at the
default `MAX_BARS = 25`:

```text
[chasm_width,
 anchor_x_0, anchor_y_0, second_x_0, second_y_0, occupied_0,
 ...]
```

Unused bar slots are zero-padded.

If you omit `options={"chasm_width": ...}` on `reset()`, the width is sampled.
`chasm_width_sampling_mode="discrete"` (the default) draws uniformly from
`ALLOWED_CHASM_WIDTHS = (8.0, 10.0, 12.0)`; `"continuous"` draws uniformly from
the closed range those widths span.

## Episode Semantics

`step(action)` validates and places one member.

- Valid placements return reward `0.0`
- Invalid placements return a small negative penalty and a reason in `info`
- `step()` does not automatically terminate the episode on success

`finalize_episode()` runs terminal structural analysis.

- disconnected graph: negative terminal reward
- mechanism / near-mechanism: negative terminal reward
- yielding / buckling: negative terminal reward
- structurally valid bridge: positive reward penalized by material use, utilization, and deflection

This split is intentional. It keeps the environment useful for:

- imitation learning from partial construction trajectories
- planning or search with an explicit stop rule
- RL setups that want to add a custom stop action wrapper later

## Support Modes

Two endpoint modes are exposed:

- `training_zones`: support anchors may slide along short horizontal start/goal zones
- `precise`: supports are fixed exactly at the chasm corners

## Public API Highlights

The main env class offers a few helpers beyond the raw Gym API:

- `preview_action_anchor(action)`: resolve snapping and validity before commit
- `resolve_anchor_query(x, y)`: inspect how a point would snap
- `snapshot_graph()`: export current nodes and bars as plain dictionaries
- `export_final_bar_sequence_anchor()`: export anchor-first actions for the realized graph
- `build_hindsight_anchor_trajectory()`: reconstruct `(obs_t, action_t)` pairs from a finished graph
- `load_hindsight_observation(obs)`: rebuild env state from a stored observation
- `build_canonical_anchor_trajectory()`: replay-check exported actions against current geometry

## Structural Model

The shipped solver is a lightweight 2D axial truss model:

- left support fixed in `x` and `y`
- right support fixed in `y`
- deck load distributed across nodes spanning the bridge
- self-weight added from member length
- member failure checked against yield and Euler buckling capacity

This is a compact research-facing solver, not a substitute for production structural design software.

## Repo Layout

```text
googym2d/
  config.py
  graph.py
  fem.py
  env.py
tests/
data/          # demonstration and correction dataset
docs/          # project website, served via GitHub Pages
```

## Dataset

`data/` bundles the human demonstrations and human-gated correction rounds used in the
papers: 667 episodes / 8,459 steps across the 8 m, 10 m, and 12 m chasm widths, stored
as Zarr v3. The observations are 126-D, matching this package's default `MAX_BARS = 25`.

The stores hold numeric arrays only and carry no participant or session identifiers.
See [data/README.md](data/README.md) for the schema, per-width breakdown, loading
example, provenance, and contributor acknowledgments.

## Versioning

**0.2.0 is a breaking change.** It aligns `MAX_BARS` (20 -> 25) and
`ALLOWED_CHASM_WIDTHS` (`(10.0,)` -> `(8.0, 10.0, 12.0)`) with the configuration
used for the published experiments. The observation dimension therefore changed
from 101 to 126, and models trained against 0.1.0 are not compatible.

This package is not on PyPI, so install a specific version straight from the tag. The
old single-width, 20-bar benchmark is tagged `v0.1.0`:

```bash
pip install "git+https://github.com/LupoSun/GooGym2D@v0.1.0"
```

## Citation

If you use this environment, please cite the relevant paper:

```bibtex
@inproceedings{sun2026teaming,
  title     = {From Teaching to Teaming: Human-AI Co-Design Environment
               for Architectural and Structural Design},
  author    = {Sun, Tao and Wang, Shaoyi and Schleicher, Simon
               and Weber, Ramon E.},
  booktitle = {Proceedings of the Association for Computer Aided Design
               in Architecture (ACADIA)},
  year      = {2026}
}

@inproceedings{sun2026capturing,
  title     = {Capturing Structural Intuition: Human-Gated Imitation
               Learning for Structural Design with Flow Matching},
  author    = {Sun, Tao and Wang, Shaoyi and Schleicher, Simon
               and Weber, Ramon E.},
  booktitle = {Proceedings of the IASS Annual Symposium},
  year      = {2026},
  organization = {International Association for Shell and Spatial Structures}
}
```

## License

MIT — see [LICENSE](LICENSE).
