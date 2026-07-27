# GooGym2D Demonstration Dataset

Human bridge-construction demonstrations and human-gated correction rounds, recorded
in the GooGym2D environment at the configuration shipped in `googym2d` 0.2.0.

All stores are **de-identified**: they contain numeric arrays only (observations,
actions, episode boundaries, terminal observations). No participant identifiers,
session identifiers, timestamps, or free-text fields are present in any array or in
the Zarr metadata.

## Contents

| Store | Episodes | Steps | Description |
|---|---:|---:|---|
| `bridge_demos.zarr` | 323 | 5,256 | Human demonstrations (browser-collected classroom demos plus previously curated 10 m demonstrations) |
| `dagger_round_001.zarr` | 78 | 808 | Human-gated corrections, DAgger round 1 |
| `dagger_round_002.zarr` | 79 | 728 | Human-gated corrections, DAgger round 2 |
| `dagger_round_003.zarr` | 100 | 882 | Human-gated corrections, DAgger round 3 |
| `dagger_round_004.zarr` | 87 | 785 | Human-gated corrections, DAgger round 4 |
| **Total** | **667** | **8,459** | |

## Schema

Zarr v3, Zstd-compressed, with arrays:

```text
data/obs          (steps, 126)   float32   observations
data/action       (steps, 4)     float32   anchor-first actions
meta/episode_ends (episodes,)    int64     exclusive end index of each episode
meta/final_obs    (episodes, 126) float32  terminal observation per episode
```

Observations follow the environment layout,
`[chasm_width, x_anchor, y_anchor, x_other, y_other, mask] * MAX_BARS`, and actions
follow `[anchor_query_x, anchor_query_y, theta_anchor, distance]`. The observation
width of 126 corresponds to `MAX_BARS = 25`.

Note that imitation-learning setups in the papers append a normalized step counter to
each observation, producing 127-D policy inputs. That extra dimension is **not** stored
here; it is added at training time.

## Chasm width coverage

Step counts per chasm width:

| Store | 8 m | 10 m | 12 m |
|---|---:|---:|---:|
| `bridge_demos.zarr` | 1,450 | 1,917 | 1,889 |
| `dagger_round_001.zarr` | 153 | 149 | 506 |
| `dagger_round_002.zarr` | 128 | 91 | 509 |
| `dagger_round_003.zarr` | 273 | 123 | 486 |
| `dagger_round_004.zarr` | 177 | 78 | 530 |

The demonstration set is close to balanced across the three widths; the correction
rounds are deliberately skewed toward 12 m, where the policy failed most often.

## Loading

Zarr is not a dependency of `googym2d` itself. Install it alongside:

```bash
pip install zarr
```

```python
import zarr
import numpy as np

store = zarr.open("data/bridge_demos.zarr", mode="r")
obs = store["data/obs"][:]           # (5256, 126)
actions = store["data/action"][:]    # (5256, 4)
ends = store["meta/episode_ends"][:] # (323,)

starts = np.concatenate([[0], ends[:-1]])
first_episode = obs[starts[0]:ends[0]]
```

## Provenance and consent

The demonstrations were contributed by students during a university architecture
design-studio session using the browser-based collector described in the papers, plus
a set of previously curated demonstrations recorded by the authors.

Participants were shown a contributor acknowledgment at the start of each session, and
every session retained in the source archive carries that acknowledgment. Identifying
fields recorded by the collector are held only in the private source archive; they were
dropped when this dataset was assembled, and none appear in the stores published here.

## Acknowledgments

We thank the following contributors, credited under the display names they entered in
the collector:

Ashlyn · Benedict Siow · bridge weaver · Chloe Lynn · June Zheng · MK · navya ·
Patrick · Phil · Qiv · Romeo · ruiqi · sunny · Wenzhuo Zhao · yak · Yuhan Hou

Names appear exactly as entered. Sessions recorded by the authors themselves, and
internal test sessions used during development, are excluded from this list.

## License and citation

The dataset is released under the same MIT license as the code. If you use it, please
cite the papers listed in the [top-level README](../README.md#citation).
