# GooGym2D

GooGym2D is a Gymnasium-based 2D bridge-construction environment extracted from the `experiments_vB` work behind the IASS/IWSS 2026 paper. This public repo intentionally contains only the core environment and structural-analysis logic needed for reinforcement learning and imitation learning research.

What is included:

- A headless `gymnasium.Env` for incremental bridge construction
- Graph-state utilities for replay, hindsight reconstruction, and inspection
- A lightweight axial truss FEM pipeline for terminal structural evaluation
- Focused tests and a small, documented Python package


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

## Environment Design

Each action attempts to place one bar in anchor-first form:

```text
[anchor_x, anchor_y, theta, length]
```

- `anchor_x, anchor_y`: where the member starts
- `theta`: direction in radians
- `length`: member length, clipped to the environment maximum

Observation is a flat vector of shape `1 + 5 * MAX_BARS`:

```text
[chasm_width,
 anchor_x_0, anchor_y_0, second_x_0, second_y_0, occupied_0,
 ...]
```

Unused bar slots are zero-padded.

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
```

