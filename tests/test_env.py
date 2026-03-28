from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

import googym2d
from googym2d import (
    BridgeBuildEnv,
    CONTINUOUS_CHASM_WIDTH_RANGE,
    ENV_ID,
    MAX_BARS,
    MAX_MEMBER_LENGTH,
    MEMBER_LENGTH,
    Node,
    Bar,
    run_fea_pipeline,
)
from googym2d.fem import (
    _active_nodes_and_bars,
    _assemble_nodal_loads,
    _build_solver_elements,
    _solve_truss,
    _support_nodes,
)
from googym2d.config import DECK_DEAD_LOAD_PER_M, DECK_LIVE_LOAD_PER_M, DEFAULT_AREA, DEFAULT_SECOND_MOMENT, DEFAULT_UNIT_WEIGHT


def anchor_action_from_endpoints(p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    assert length <= MAX_MEMBER_LENGTH + 1e-6
    theta = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    return np.array([p0[0], p0[1], theta, length], dtype=np.float32)


def make_env(
    chasm_width: float = 10.0,
    *,
    endpoint_mode: str = "training_zones",
    chasm_width_sampling_mode: str = "discrete",
) -> BridgeBuildEnv:
    env = BridgeBuildEnv(
        endpoint_mode=endpoint_mode,
        chasm_width_sampling_mode=chasm_width_sampling_mode,
    )
    env.reset(seed=0, options={"chasm_width": chasm_width})
    return env


def make_manual_graph(
    node_specs: list[tuple[int, float, float, str, bool]],
    bar_specs: list[tuple[int, int]],
) -> tuple[dict[int, Node], dict[int, Bar]]:
    nodes = {
        node_id: Node(id=node_id, x=float(x), y=float(y), kind=kind, movable=bool(movable))
        for node_id, x, y, kind, movable in node_specs
    }
    bars: dict[int, Bar] = {}
    for bar_id, (node_u, node_v) in enumerate(bar_specs, start=1):
        bar = Bar(id=bar_id, node_u=node_u, node_v=node_v, placement_order=bar_id - 1)
        bars[bar_id] = bar
        nodes[node_u].incident_bar_ids.add(bar_id)
        nodes[node_v].incident_bar_ids.add(bar_id)
    return nodes, bars


def test_registration_and_reset_shape() -> None:
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert obs.shape == (1 + 5 * MAX_BARS,)
    assert "has_span" in info
    env.close()


def test_support_reuse_and_duplicate_bar_rejection() -> None:
    env = make_env()

    obs, reward, terminated, truncated, info = env.step(anchor_action_from_endpoints((0.0, 0.0), (3.0, 0.0)))
    assert obs.shape == (1 + 5 * MAX_BARS,)
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["outcome"] == "placed"

    diagonal_end = (1.5, math.sqrt(MEMBER_LENGTH * MEMBER_LENGTH - 1.5 * 1.5))
    _obs, reward, _terminated, _truncated, info = env.step(anchor_action_from_endpoints((0.0, 0.0), diagonal_end))
    assert reward == 0.0
    assert info["outcome"] == "placed"
    left_supports = [node for node in env.nodes.values() if node.kind == "left_support"]
    assert len(left_supports) == 1

    _obs, reward, _terminated, _truncated, info = env.step(anchor_action_from_endpoints((0.0, 0.0), (3.0, 0.0)))
    assert reward < 0.0
    assert info["outcome"] == "invalid"
    assert info["reason"] == "duplicate_bar"


def test_training_zones_allow_top_surface_supports_beyond_corners() -> None:
    env = make_env(6.0, endpoint_mode="training_zones")

    left_resolution = env.resolve_anchor_query(-1.5, 0.3)
    right_resolution = env.resolve_anchor_query(7.6, 0.4)
    assert left_resolution is not None
    assert right_resolution is not None
    assert left_resolution.support_kind == "left_support"
    assert right_resolution.support_kind == "right_support"
    np.testing.assert_allclose(left_resolution.point, [-1.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(right_resolution.point, [7.6, 0.0], atol=1e-6)


def test_hindsight_round_trip_and_replay() -> None:
    env = make_env(6.0, endpoint_mode="precise")
    env.step(anchor_action_from_endpoints((0.0, 0.0), (3.0, 0.0)))
    env.step(anchor_action_from_endpoints((3.0, 0.0), (6.0, 0.0)))

    hindsight_obs, hindsight_actions = env.build_hindsight_anchor_trajectory()
    assert len(hindsight_obs) == 2
    assert len(hindsight_actions) == 2

    replay_env = make_env(6.0, endpoint_mode="precise")
    np.testing.assert_allclose(replay_env.load_hindsight_observation(hindsight_obs[1]), hindsight_obs[1], atol=1e-6)

    replayable, reason = env.can_replay_exported_anchor_sequence()
    assert replayable
    assert reason is None


def test_continuous_width_sampling_mode_supports_non_discrete_widths() -> None:
    env = BridgeBuildEnv(chasm_width_sampling_mode="continuous")
    low, high = CONTINUOUS_CHASM_WIDTH_RANGE
    sampled_widths: list[float] = []
    for seed in range(8):
        _obs, _info = env.reset(seed=seed)
        sampled_widths.append(float(env.chasm_width))
        assert low <= env.chasm_width <= high

    if math.isclose(low, high, abs_tol=1e-9):
        assert all(math.isclose(width, low, abs_tol=1e-6) for width in sampled_widths)


def test_fem_pipeline_and_finalize_success_and_disconnected() -> None:
    env = make_env(6.0, endpoint_mode="precise")
    left = next(node for node in env.nodes.values() if node.kind == "left_support")
    right = next(node for node in env.nodes.values() if node.kind == "right_support")
    mid = env._add_node(3.0, 0.0, kind="free", movable=True)
    top = env._add_node(3.0, 2.0, kind="free", movable=True)
    env._add_bar(left.id, mid.id, placement_order=0)
    env._add_bar(mid.id, right.id, placement_order=1)
    env._add_bar(left.id, top.id, placement_order=2)
    env._add_bar(top.id, right.id, placement_order=3)
    env._add_bar(top.id, mid.id, placement_order=4)

    result = run_fea_pipeline(env.nodes, env.bars, env.chasm_width)
    assert result.solver_ok
    assert result.status == "ok"
    assert result.max_utilization < 1.0

    _obs, reward, terminated, truncated, info = env.finalize_episode()
    assert terminated and not truncated
    assert info["outcome"] == "success"
    assert reward > 0.0

    disconnected = make_env(6.0, endpoint_mode="precise")
    left = next(node for node in disconnected.nodes.values() if node.kind == "left_support")
    mid = disconnected._add_node(3.0, 0.0, kind="free", movable=True)
    disconnected._add_bar(left.id, mid.id, placement_order=0)
    _obs, reward, terminated, truncated, info = disconnected.finalize_episode()
    assert terminated and not truncated
    assert reward == -10.0
    assert info["outcome"] == "disconnected"


def test_triangle_loads_and_reactions_are_finite() -> None:
    nodes, bars = make_manual_graph(
        [
            (1, 0.0, 0.0, "left_support", False),
            (2, 10.0, 0.0, "right_support", False),
            (3, 5.0, 3.0, "free", True),
        ],
        [(1, 2), (1, 3), (3, 2)],
    )

    active_nodes, active_bars = _active_nodes_and_bars(nodes, bars)
    load_distribution = _assemble_nodal_loads(
        active_nodes,
        active_bars,
        10.0,
        deck_line_load=DECK_DEAD_LOAD_PER_M + DECK_LIVE_LOAD_PER_M,
        area=DEFAULT_AREA,
        unit_weight=DEFAULT_UNIT_WEIGHT,
    )
    solver_nodes, solver_elements = _build_solver_elements(active_nodes, active_bars)
    solve_result = _solve_truss(
        solver_nodes,
        solver_elements,
        _support_nodes(active_nodes),
        load_distribution.nodal_loads,
        area=DEFAULT_AREA,
        second_moment=DEFAULT_SECOND_MOMENT,
        young_modulus=210e9,
        yield_strength=355e6,
    )

    assert solve_result.solver_ok
    assert math.isfinite(solve_result.max_displacement)
    left_reaction = solve_result.support_reactions[1]
    right_reaction = solve_result.support_reactions[2]
    assert math.isclose(left_reaction[1], right_reaction[1], rel_tol=1e-6, abs_tol=1e-6)


def test_mechanism_and_buckling_classifications() -> None:
    nodes, bars = make_manual_graph(
        [
            (1, 0.0, 0.0, "left_support", False),
            (2, 10.0, 0.0, "right_support", False),
            (3, 5.0, 3.0, "free", True),
        ],
        [(1, 3), (3, 2)],
    )
    result = run_fea_pipeline(nodes, bars, 10.0)
    assert not result.solver_ok
    assert result.status == "mechanism"

    stable_nodes, stable_bars = make_manual_graph(
        [
            (1, 0.0, 0.0, "left_support", False),
            (2, 10.0, 0.0, "right_support", False),
            (3, 5.0, 3.0, "free", True),
        ],
        [(1, 2), (1, 3), (3, 2)],
    )
    buckling = run_fea_pipeline(stable_nodes, stable_bars, 10.0, second_moment=1e-9)
    assert buckling.solver_ok
    assert buckling.status == "buckling"
    assert buckling.max_utilization > 1.0

