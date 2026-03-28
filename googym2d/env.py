"""Gymnasium environment for graph-based bridge construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import gymnasium
import numpy as np
from gymnasium import spaces
from shapely.geometry import LineString, Point, box

from googym2d.config import (
    ALLOWED_CHASM_WIDTHS,
    CLIFF_HEIGHT,
    CLIFF_WIDTH,
    CONTINUOUS_CHASM_WIDTH_RANGE,
    CORNER_EQUIVALENCE_TOL,
    DEFAULT_CHASM_WIDTH_SAMPLING_MODE,
    DEFLECTION_LIMIT,
    INVALID_ACTION_PENALTY,
    MAX_BARS,
    MAX_MEMBER_LENGTH,
    MEMBER_LENGTH,
    MERGE_EPS,
    MIN_MEMBER_LENGTH,
    NODE_SNAP_RADIUS,
    SECONDARY_DIRECTION_SNAP_RAD,
    SUPPORT_SNAP_RADIUS,
    TRAIN_GOAL_ZONE_WIDTH,
    TRAIN_START_ZONE_WIDTH,
)
from googym2d.fem import run_fea_pipeline
from googym2d.graph import Bar, FemResult, Node, bar_endpoints, bar_geometry, bar_oriented_endpoints, snapshot_graph


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass
class EndpointResolution:
    kind: str
    node_id: int | None
    point: tuple[float, float]
    distance: float
    support_kind: str | None = None


@dataclass
class PlacementPreview:
    valid: bool
    reason: str | None
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    anchor_endpoint_index: int | None
    anchor_resolution: EndpointResolution | None
    second_resolution: EndpointResolution | None
    total_snap_distance: float


EndpointMode = Literal["training_zones", "precise"]
ChasmWidthSamplingMode = Literal["discrete", "continuous"]


class BridgeBuildEnv(gymnasium.Env):
    """Headless Gymnasium env for bridge construction.

    The environment separates incremental graph editing from structural
    evaluation. Each `step(action)` tries to place one truss member. Terminal
    FEM evaluation is triggered explicitly with `finalize_episode()`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        endpoint_mode: EndpointMode = "training_zones",
        chasm_width_sampling_mode: ChasmWidthSamplingMode = DEFAULT_CHASM_WIDTH_SAMPLING_MODE,
    ):
        super().__init__()
        if render_mode is not None:
            raise ValueError("This public package is headless. GUI rendering is intentionally omitted.")
        if endpoint_mode not in {"training_zones", "precise"}:
            raise ValueError(f"Unsupported endpoint_mode: {endpoint_mode}")
        if chasm_width_sampling_mode not in {"discrete", "continuous"}:
            raise ValueError(f"Unsupported chasm_width_sampling_mode: {chasm_width_sampling_mode}")

        self.render_mode = None
        self.endpoint_mode: EndpointMode = endpoint_mode
        self.chasm_width_sampling_mode: ChasmWidthSamplingMode = chasm_width_sampling_mode
        self.max_bars = MAX_BARS

        obs_dim = 1 + 5 * self.max_bars
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-5.0, -10.0, -math.pi, 0.0], dtype=np.float32),
            high=np.array([50.0, 20.0, math.pi, MAX_MEMBER_LENGTH], dtype=np.float32),
            dtype=np.float32,
        )

        self.chasm_width: float = 0.0
        self.left_cliff = None
        self.right_cliff = None
        self.nodes: dict[int, Node] = {}
        self.bars: dict[int, Bar] = {}
        self.preview_angle: float = 0.0
        self.hovered_node: int | None = None
        self.selected_node: int | None = None
        self.current_preview_anchor: int | None = None
        self.finalized = False
        self.fem_result: FemResult | None = None
        self.last_preview: PlacementPreview | None = None
        self._next_node_id = 1
        self._next_bar_id = 1

    def left_anchor_point(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def right_anchor_point(self) -> tuple[float, float]:
        return (self.chasm_width, 0.0)

    def left_start_zone(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_min = max(-CLIFF_WIDTH, -TRAIN_START_ZONE_WIDTH)
        return (x_min, 0.0), (0.0, 0.0)

    def right_goal_zone(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_max = self.chasm_width + min(CLIFF_WIDTH, TRAIN_GOAL_ZONE_WIDTH)
        return (self.chasm_width, 0.0), (x_max, 0.0)

    @property
    def num_active_bars(self) -> int:
        return sum(1 for bar in self.bars.values() if bar.active)

    @property
    def has_span(self) -> bool:
        return self._has_spanning_component() and self._supports_well_braced()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and "chasm_width" in options:
            self.chasm_width = float(options["chasm_width"])
        elif self.chasm_width_sampling_mode == "continuous":
            low, high = CONTINUOUS_CHASM_WIDTH_RANGE
            self.chasm_width = float(self.np_random.uniform(low, high))
        else:
            self.chasm_width = float(self.np_random.choice(np.asarray(ALLOWED_CHASM_WIDTHS, dtype=np.float32)))

        self.left_cliff = box(-CLIFF_WIDTH, -CLIFF_HEIGHT, 0.0, 0.0)
        self.right_cliff = box(self.chasm_width, -CLIFF_HEIGHT, self.chasm_width + CLIFF_WIDTH, 0.0)
        self.nodes = {}
        self.bars = {}
        self.preview_angle = 0.0
        self.hovered_node = None
        self.selected_node = None
        self.current_preview_anchor = None
        self.finalized = False
        self.fem_result = None
        self.last_preview = None
        self._next_node_id = 1
        self._next_bar_id = 1

        if self.endpoint_mode == "precise":
            self._find_or_create_support_node(self.left_anchor_point(), "left_support")
            self._find_or_create_support_node(self.right_anchor_point(), "right_support")

        return self._build_observation(), {"has_span": self.has_span}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.finalized:
            return self._build_observation(), INVALID_ACTION_PENALTY, False, False, {
                "outcome": "invalid",
                "reason": "episode_finalized",
            }
        if self.num_active_bars >= self.max_bars:
            return self._build_observation(), INVALID_ACTION_PENALTY, False, False, {
                "outcome": "invalid",
                "reason": "max_bars_reached",
            }

        preview = self.preview_action_anchor(action)
        if not preview.valid or preview.anchor_resolution is None or preview.second_resolution is None:
            return self._build_observation(), INVALID_ACTION_PENALTY, False, False, {
                "outcome": "invalid",
                "reason": preview.reason or "unresolved_action",
                **self._preview_metadata(preview),
            }

        pre_snapshot = self.snapshot_graph()
        bar = self._commit_preview(preview)
        obs = self._build_observation()
        return obs, 0.0, False, False, {
            "outcome": "placed",
            "bar_id": bar.id,
            "has_span": self.has_span,
            "pre_graph": pre_snapshot,
            "post_graph": self.snapshot_graph(),
            **self._preview_metadata(preview),
        }

    def finalize_episode(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.finalized = True
        obs = self._build_observation()
        if not self.has_span:
            self.fem_result = None
            return obs, -10.0, True, False, {
                "outcome": "disconnected",
                "has_span": False,
                "finalized": True,
            }

        self.fem_result = run_fea_pipeline(self.nodes, self.bars, self.chasm_width)
        result = self.fem_result
        info = {
            "outcome": "success",
            "has_span": True,
            "finalized": True,
            **result.as_dict(),
        }
        if not result.solver_ok or result.status in {"mechanism", "near_mechanism"}:
            info["outcome"] = result.status
            reward = -15.0
        elif result.status in {"yield", "buckling"}:
            info["outcome"] = result.status
            reward = -20.0
        else:
            material_penalty = 0.5 * self.num_active_bars
            utilization_penalty = 10.0 * result.max_utilization
            deflection_ratio = min(1.0, result.max_displacement / max(DEFLECTION_LIMIT, 1e-9))
            deflection_penalty = 5.0 * deflection_ratio
            reward = 50.0 - material_penalty - utilization_penalty - deflection_penalty
        return obs, float(reward), True, False, info

    def preview_action(self, action: np.ndarray | list[float] | tuple[float, ...]) -> PlacementPreview:
        return self.preview_action_midpoint(action)

    def preview_action_midpoint(
        self,
        action: np.ndarray | list[float] | tuple[float, float, float],
    ) -> PlacementPreview:
        action_arr = np.asarray(action, dtype=np.float32)
        x, y, theta = float(action_arr[0]), float(action_arr[1]), float(action_arr[2])
        self.preview_angle = theta
        raw_endpoints = self._midpoint_action_to_endpoints(x, y, theta)

        candidates: list[PlacementPreview] = []
        for anchor_index in (0, 1):
            anchor_raw = raw_endpoints[anchor_index]
            other_raw = raw_endpoints[1 - anchor_index]
            anchor_resolution = self._resolve_anchor(anchor_raw)
            if anchor_resolution is None:
                continue
            second_resolution = self._resolve_secondary(other_raw, anchor_resolution=anchor_resolution)
            if second_resolution is None:
                continue
            endpoints = [anchor_resolution.point, second_resolution.point]
            valid, reason = self._validate_candidate(anchor_resolution, second_resolution, tuple(endpoints))
            candidates.append(
                PlacementPreview(
                    valid=valid,
                    reason=reason,
                    endpoints=(tuple(endpoints[0]), tuple(endpoints[1])),
                    anchor_endpoint_index=anchor_index,
                    anchor_resolution=anchor_resolution,
                    second_resolution=second_resolution,
                    total_snap_distance=anchor_resolution.distance + second_resolution.distance,
                )
            )

        if not candidates:
            self.last_preview = PlacementPreview(
                valid=False,
                reason="anchor_not_found",
                endpoints=raw_endpoints,
                anchor_endpoint_index=None,
                anchor_resolution=None,
                second_resolution=None,
                total_snap_distance=float("inf"),
            )
            self.hovered_node = None
            self.current_preview_anchor = None
            return self.last_preview

        candidates.sort(key=lambda item: (not item.valid, item.total_snap_distance))
        return self._set_preview(candidates[0])

    def preview_action_anchor(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> PlacementPreview:
        anchor_x, anchor_y, theta_anchor, distance = self._parse_anchor_action(action)
        self.preview_angle = theta_anchor

        raw_second = (
            anchor_x + distance * math.cos(theta_anchor),
            anchor_y + distance * math.sin(theta_anchor),
        )
        anchor_resolution = self._resolve_anchor((anchor_x, anchor_y))
        if anchor_resolution is None:
            return self._set_preview(
                PlacementPreview(
                    valid=False,
                    reason="anchor_not_found",
                    endpoints=((anchor_x, anchor_y), raw_second),
                    anchor_endpoint_index=None,
                    anchor_resolution=None,
                    second_resolution=None,
                    total_snap_distance=float("inf"),
                )
            )

        resolved_second_raw = (
            anchor_resolution.point[0] + distance * math.cos(theta_anchor),
            anchor_resolution.point[1] + distance * math.sin(theta_anchor),
        )
        second_resolution = self._resolve_secondary(resolved_second_raw, anchor_resolution=anchor_resolution)
        endpoints = (anchor_resolution.point, second_resolution.point)
        valid, reason = self._validate_candidate(anchor_resolution, second_resolution, endpoints)
        return self._set_preview(
            PlacementPreview(
                valid=valid,
                reason=reason,
                endpoints=endpoints,
                anchor_endpoint_index=0,
                anchor_resolution=anchor_resolution,
                second_resolution=second_resolution,
                total_snap_distance=anchor_resolution.distance + second_resolution.distance,
            )
        )

    def build_human_action(self, x: float, y: float, theta: float) -> np.ndarray:
        anchor_resolution = self._resolve_anchor((float(x), float(y)))
        if anchor_resolution is None:
            return np.array([x, y, theta], dtype=np.float32)

        end_x = anchor_resolution.point[0] + MEMBER_LENGTH * math.cos(theta)
        end_y = anchor_resolution.point[1] + MEMBER_LENGTH * math.sin(theta)
        mid_x = 0.5 * (anchor_resolution.point[0] + end_x)
        mid_y = 0.5 * (anchor_resolution.point[1] + end_y)
        return np.array([mid_x, mid_y, theta], dtype=np.float32)

    def build_human_anchor_action(self, x: float, y: float, theta: float, distance: float) -> np.ndarray:
        return np.array([x, y, theta, distance], dtype=np.float32)

    def midpoint_action_to_anchor_action(
        self,
        action: np.ndarray | list[float] | tuple[float, float, float],
    ) -> np.ndarray:
        preview = self.preview_action_midpoint(action)
        if not preview.valid or preview.anchor_resolution is None or preview.anchor_endpoint_index is None:
            raise ValueError("Cannot convert invalid midpoint action into anchor form.")

        theta_mid = float(np.asarray(action, dtype=np.float32)[2])
        theta_anchor = theta_mid
        if preview.anchor_endpoint_index == 1:
            theta_anchor = wrap_angle(theta_mid + math.pi)
        anchor_point = preview.anchor_resolution.point
        second_point = preview.second_resolution.point
        distance = math.hypot(second_point[0] - anchor_point[0], second_point[1] - anchor_point[1])
        return np.array([anchor_point[0], anchor_point[1], theta_anchor, distance], dtype=np.float32)

    def export_final_bar_sequence(self) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        for bar in self._sorted_active_bars():
            mid_x, mid_y, theta, _length = bar_geometry(bar, self.nodes)
            actions.append(np.array([mid_x, mid_y, theta], dtype=np.float32))
        return actions

    def export_final_bar_sequence_anchor(self) -> list[np.ndarray]:
        actions: list[np.ndarray] = []
        for bar in self._sorted_active_bars():
            anchor_point, second_point = bar_oriented_endpoints(bar, self.nodes)
            actions.append(self._anchor_action_from_endpoints(anchor_point, second_point))
        return actions

    def build_hindsight_anchor_trajectory(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        bars = self._sorted_active_bars()
        obs_seq: list[np.ndarray] = []
        action_seq: list[np.ndarray] = []
        for idx, bar in enumerate(bars):
            obs_seq.append(self._build_observation_from_bars(bars[:idx]))
            anchor_point, second_point = bar_oriented_endpoints(bar, self.nodes)
            action_seq.append(self._anchor_action_from_endpoints(anchor_point, second_point))
        return obs_seq, action_seq

    def load_hindsight_observation(self, observation: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(observation, dtype=np.float32)
        self.reset(options={"chasm_width": float(obs_arr[0])})

        node_ids_by_point: list[tuple[tuple[float, float], int]] = []

        def classify_kind(point: tuple[float, float]) -> tuple[str, bool]:
            support_kind = self._support_kind_for_point(point)
            if support_kind is not None:
                return support_kind, False
            return "free", True

        def find_or_create_node(point: tuple[float, float]) -> int:
            for existing_point, node_id in node_ids_by_point:
                if self._points_match(existing_point, point):
                    return node_id
            kind, movable = classify_kind(point)
            if kind in {"left_support", "right_support"}:
                node = self._find_or_create_support_node(point, kind)
            else:
                node = self._add_node(point[0], point[1], kind=kind, movable=movable)
            node_ids_by_point.append((point, node.id))
            return node.id

        for idx in range(self.max_bars):
            base = 1 + 5 * idx
            if base + 4 >= obs_arr.shape[0] or obs_arr[base + 4] < 0.5:
                break
            anchor_point = (float(obs_arr[base]), float(obs_arr[base + 1]))
            second_point = (float(obs_arr[base + 2]), float(obs_arr[base + 3]))
            node_u = find_or_create_node(anchor_point)
            node_v = find_or_create_node(second_point)
            self._add_bar(node_u, node_v, placement_order=idx)

        self.preview_angle = 0.0
        self.hovered_node = None
        self.selected_node = None
        self.current_preview_anchor = None
        self.finalized = False
        self.fem_result = None
        self.last_preview = None
        return self._build_observation()

    def build_canonical_anchor_trajectory(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], str | None]:
        replay_env = BridgeBuildEnv(endpoint_mode=self.endpoint_mode)
        canonical_actions = [np.array(action, dtype=np.float32) for action in self.export_final_bar_sequence_anchor()]
        canonical_obs: list[np.ndarray] = []
        try:
            obs, _info = replay_env.reset(options={"chasm_width": self.chasm_width})
            for step_idx, action in enumerate(canonical_actions, start=1):
                canonical_obs.append(obs.copy())
                obs, _reward, _terminated, _truncated, info = replay_env.step(action)
                if info.get("outcome") != "placed":
                    reason = info.get("reason") or info.get("outcome") or "replay_failed"
                    return [], [], f"replay_step_{step_idx}_{reason}"
            if not self._matches_active_bar_geometry(replay_env):
                return [], [], "graph_mismatch"
            return canonical_obs, canonical_actions, None
        finally:
            replay_env.close()

    def can_replay_exported_anchor_sequence(self) -> tuple[bool, str | None]:
        _canonical_obs, _canonical_actions, reason = self.build_canonical_anchor_trajectory()
        return reason is None, reason

    def snapshot_graph(self) -> dict[str, Any]:
        return snapshot_graph(self.nodes, self.bars)

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _set_preview(self, preview: PlacementPreview) -> PlacementPreview:
        self.last_preview = preview
        if preview.anchor_resolution is not None:
            self.current_preview_anchor = preview.anchor_resolution.node_id
            self.hovered_node = preview.anchor_resolution.node_id
        else:
            self.current_preview_anchor = None
            self.hovered_node = None
        return preview

    def _node_resolution(self, node: Node) -> EndpointResolution:
        if node.kind == "free":
            return EndpointResolution("node", node.id, (float(node.x), float(node.y)), 0.0)
        return EndpointResolution("support", node.id, (float(node.x), float(node.y)), 0.0, support_kind=node.kind)

    @staticmethod
    def _anchor_action_from_endpoints(
        anchor_point: tuple[float, float],
        second_point: tuple[float, float],
    ) -> np.ndarray:
        distance = math.hypot(second_point[0] - anchor_point[0], second_point[1] - anchor_point[1])
        theta_anchor = wrap_angle(math.atan2(second_point[1] - anchor_point[1], second_point[0] - anchor_point[0]))
        return np.array([anchor_point[0], anchor_point[1], theta_anchor, distance], dtype=np.float32)

    def _preview_metadata(self, preview: PlacementPreview | None) -> dict[str, Any]:
        if preview is None or preview.anchor_resolution is None or preview.second_resolution is None:
            return {}

        anchor_point, second_point = preview.endpoints
        distance = math.hypot(second_point[0] - anchor_point[0], second_point[1] - anchor_point[1])
        theta_anchor = float(self.preview_angle) if distance <= MERGE_EPS else float(
            self._anchor_action_from_endpoints(anchor_point, second_point)[2]
        )
        return {
            "anchor_point": [float(anchor_point[0]), float(anchor_point[1])],
            "second_point": [float(second_point[0]), float(second_point[1])],
            "anchor_kind": preview.anchor_resolution.kind,
            "anchor_node_id": preview.anchor_resolution.node_id,
            "anchor_support_kind": preview.anchor_resolution.support_kind,
            "second_kind": preview.second_resolution.kind,
            "second_node_id": preview.second_resolution.node_id,
            "second_support_kind": preview.second_resolution.support_kind,
            "theta_anchor": theta_anchor,
            "distance": float(distance),
            "action_anchor": [
                float(anchor_point[0]),
                float(anchor_point[1]),
                theta_anchor,
                float(distance),
            ],
            "total_snap_distance": float(preview.total_snap_distance),
        }

    def resolve_anchor_query(self, x: float, y: float) -> EndpointResolution | None:
        return self._resolve_anchor((float(x), float(y)))

    def _support_kind_for_point(self, point: tuple[float, float]) -> str | None:
        if self._points_match(point, self.left_anchor_point()):
            return "left_support"
        if self._points_match(point, self.right_anchor_point()):
            return "right_support"
        if self._point_on_horizontal_support(point, *self.left_start_zone()):
            return "left_support"
        if self._point_on_horizontal_support(point, *self.right_goal_zone()):
            return "right_support"
        return None

    def _matches_active_bar_geometry(self, other: BridgeBuildEnv) -> bool:
        self_bars = self._sorted_active_bars()
        other_bars = other._sorted_active_bars()
        if len(self_bars) != len(other_bars):
            return False
        for self_bar, other_bar in zip(self_bars, other_bars, strict=True):
            self_anchor, self_second = bar_oriented_endpoints(self_bar, self.nodes)
            other_anchor, other_second = bar_oriented_endpoints(other_bar, other.nodes)
            if not self._points_match(self_anchor, other_anchor):
                return False
            if not self._points_match(self_second, other_second):
                return False
            if self.nodes[self_bar.node_u].kind != other.nodes[other_bar.node_u].kind:
                return False
            if self.nodes[self_bar.node_v].kind != other.nodes[other_bar.node_v].kind:
                return False
        return True

    def _commit_preview(self, preview: PlacementPreview) -> Bar:
        assert preview.anchor_resolution is not None
        assert preview.second_resolution is not None
        node_u = self._materialize_resolution(preview.anchor_resolution)
        node_v = self._materialize_resolution(preview.second_resolution)
        if node_u.id == node_v.id:
            raise RuntimeError("Resolved identical endpoints for committed preview.")
        bar = self._add_bar(node_u.id, node_v.id, placement_order=self.num_active_bars)
        self.last_preview = preview
        return bar

    def _materialize_resolution(self, resolution: EndpointResolution) -> Node:
        if resolution.node_id is not None and resolution.node_id in self.nodes:
            return self.nodes[resolution.node_id]
        point = resolution.point
        if resolution.kind == "support":
            assert resolution.support_kind is not None
            return self._find_or_create_support_node(point, resolution.support_kind)
        return self._add_node(point[0], point[1], kind="free", movable=True)

    def _resolve_anchor(self, point: tuple[float, float]) -> EndpointResolution | None:
        candidates: list[EndpointResolution] = []
        node_candidate = self._nearest_existing_node(point, NODE_SNAP_RADIUS)
        if node_candidate is not None:
            node, distance = node_candidate
            candidates.append(EndpointResolution("node", node.id, (float(node.x), float(node.y)), distance))
        support_candidate = self._nearest_support_snap(point)
        if support_candidate is not None:
            support_kind, snapped_point, distance = support_candidate
            existing_support = self._find_existing_support(snapped_point, support_kind)
            candidates.append(
                EndpointResolution(
                    "support",
                    None if existing_support is None else existing_support.id,
                    snapped_point,
                    distance,
                    support_kind=support_kind,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.distance)

    def _resolve_secondary(
        self,
        point: tuple[float, float],
        *,
        anchor_resolution: EndpointResolution | None = None,
    ) -> EndpointResolution:
        candidates: list[EndpointResolution] = []
        node_candidate = self._nearest_existing_node(point, NODE_SNAP_RADIUS)
        if node_candidate is not None:
            node, distance = node_candidate
            candidates.append(EndpointResolution("node", node.id, (float(node.x), float(node.y)), distance))
        support_candidate = self._nearest_support_snap(point)
        if support_candidate is not None:
            support_kind, snapped_point, distance = support_candidate
            existing_support = self._find_existing_support(snapped_point, support_kind)
            candidates.append(
                EndpointResolution(
                    "support",
                    None if existing_support is None else existing_support.id,
                    snapped_point,
                    distance,
                    support_kind=support_kind,
                )
            )
        if candidates:
            candidates.sort(key=lambda item: item.distance)
            return candidates[0]
        if anchor_resolution is not None:
            directional_candidate = self._resolve_secondary_along_anchor_direction(anchor_resolution, point)
            if directional_candidate is not None:
                return directional_candidate
        return EndpointResolution("free", None, point, 0.0)

    def _resolve_secondary_along_anchor_direction(
        self,
        anchor_resolution: EndpointResolution,
        point: tuple[float, float],
    ) -> EndpointResolution | None:
        anchor_point = anchor_resolution.point
        intended_dx = point[0] - anchor_point[0]
        intended_dy = point[1] - anchor_point[1]
        intended_length = math.hypot(intended_dx, intended_dy)
        if intended_length <= MERGE_EPS:
            return None

        intended_theta = math.atan2(intended_dy, intended_dx)
        best: tuple[tuple[float, float, float], EndpointResolution] | None = None
        for node in self.nodes.values():
            if anchor_resolution.node_id is not None and node.id == anchor_resolution.node_id:
                continue
            candidate_point = (float(node.x), float(node.y))
            candidate_dx = candidate_point[0] - anchor_point[0]
            candidate_dy = candidate_point[1] - anchor_point[1]
            candidate_length = math.hypot(candidate_dx, candidate_dy)
            if candidate_length < MIN_MEMBER_LENGTH or candidate_length > MAX_MEMBER_LENGTH:
                continue

            angle_error = abs(wrap_angle(math.atan2(candidate_dy, candidate_dx) - intended_theta))
            if angle_error > SECONDARY_DIRECTION_SNAP_RAD:
                continue

            radial_error = abs(candidate_length - intended_length)
            distance_score = angle_error + 0.05 * radial_error
            if node.kind in {"left_support", "right_support"}:
                resolution = EndpointResolution("support", node.id, candidate_point, distance_score, support_kind=node.kind)
            else:
                resolution = EndpointResolution("node", node.id, candidate_point, distance_score)

            score = (distance_score, radial_error, angle_error)
            if best is None or score < best[0]:
                best = (score, resolution)
        if best is None:
            return None
        return best[1]

    def _validate_candidate(
        self,
        anchor_resolution: EndpointResolution,
        second_resolution: EndpointResolution,
        endpoints: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[bool, str | None]:
        p0, p1 = endpoints
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= MERGE_EPS:
            return False, "coincident_endpoints"

        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length < MIN_MEMBER_LENGTH:
            return False, "member_too_short"
        if length > MAX_MEMBER_LENGTH:
            return False, "member_too_long"

        node_u_id = anchor_resolution.node_id
        node_v_id = second_resolution.node_id
        if node_u_id is not None and node_u_id == node_v_id:
            return False, "duplicate_endpoint_node"
        if self._resolutions_materialize_to_same_node(anchor_resolution, second_resolution):
            return False, "duplicate_endpoint_node"

        if second_resolution.kind == "free":
            for node in self.nodes.values():
                if math.hypot(node.x - p1[0], node.y - p1[1]) <= MERGE_EPS:
                    return False, "free_node_collides_existing"

        if self._bar_exists_between(node_u_id, node_v_id, p0, p1):
            return False, "duplicate_bar"
        if self._has_exact_overlap(p0, p1):
            return False, "exact_overlap"
        if self._segment_hits_cliff_interior(p0, p1, anchor_resolution, second_resolution):
            return False, "cliff_intersection"
        return True, None

    def _resolution_materialization_key(self, resolution: EndpointResolution) -> tuple[Any, ...]:
        if resolution.node_id is not None:
            return ("existing", int(resolution.node_id))
        if resolution.kind == "support":
            assert resolution.support_kind is not None
            if self.endpoint_mode == "training_zones":
                canonical = self._canonical_support_node(resolution.support_kind)
                if canonical is not None:
                    return ("existing", int(canonical.id))
                return ("pending_canonical_support", resolution.support_kind)
            existing = self._find_existing_support(resolution.point, resolution.support_kind)
            if existing is not None:
                return ("existing", int(existing.id))
            return (
                "pending_support",
                resolution.support_kind,
                round(float(resolution.point[0]), 6),
                round(float(resolution.point[1]), 6),
            )
        return (
            "pending_free",
            round(float(resolution.point[0]), 6),
            round(float(resolution.point[1]), 6),
        )

    def _resolutions_materialize_to_same_node(
        self,
        anchor_resolution: EndpointResolution,
        second_resolution: EndpointResolution,
    ) -> bool:
        return self._resolution_materialization_key(anchor_resolution) == self._resolution_materialization_key(second_resolution)

    def _nearest_existing_node(self, point: tuple[float, float], radius: float) -> tuple[Node, float] | None:
        best: tuple[Node, float] | None = None
        for node in self.nodes.values():
            distance = math.hypot(node.x - point[0], node.y - point[1])
            if distance <= radius and (best is None or distance < best[1]):
                best = (node, distance)
        return best

    def _nearest_support_snap(self, point: tuple[float, float]) -> tuple[str, tuple[float, float], float] | None:
        candidates = self._support_snap_candidates(point)
        best = None
        for support_kind, snapped_point in candidates:
            distance = math.hypot(snapped_point[0] - point[0], snapped_point[1] - point[1])
            if distance <= SUPPORT_SNAP_RADIUS and (best is None or distance < best[2]):
                best = (support_kind, snapped_point, distance)
        return best

    def _canonical_support_node(self, support_kind: str) -> Node | None:
        candidates = [node for node in self.nodes.values() if node.kind == support_kind]
        if not candidates:
            return None
        connected = [node for node in candidates if node.incident_bar_ids]
        pool = connected or candidates
        return min(pool, key=lambda node: node.id)

    def _find_existing_support(self, point: tuple[float, float], support_kind: str) -> Node | None:
        for node in self.nodes.values():
            if node.kind != support_kind:
                continue
            if math.hypot(node.x - point[0], node.y - point[1]) <= MERGE_EPS:
                return node
        return None

    def _find_or_create_support_node(self, point: tuple[float, float], support_kind: str) -> Node:
        if self.endpoint_mode == "training_zones":
            canonical = self._canonical_support_node(support_kind)
            if canonical is not None:
                return canonical
        existing = self._find_existing_support(point, support_kind)
        if existing is not None:
            return existing
        return self._add_node(point[0], point[1], kind=support_kind, movable=False)

    def _support_snap_candidates(self, point: tuple[float, float]) -> list[tuple[str, tuple[float, float]]]:
        if self.endpoint_mode == "precise":
            return [("left_support", self.left_anchor_point()), ("right_support", self.right_anchor_point())]
        return [
            (
                "left_support",
                (
                    (float(existing_left.x), float(existing_left.y))
                    if (existing_left := self._canonical_support_node("left_support")) is not None
                    else self._project_point_to_horizontal_support(point, *self.left_start_zone())
                ),
            ),
            (
                "right_support",
                (
                    (float(existing_right.x), float(existing_right.y))
                    if (existing_right := self._canonical_support_node("right_support")) is not None
                    else self._project_point_to_horizontal_support(point, *self.right_goal_zone())
                ),
            ),
        ]

    @staticmethod
    def _project_point_to_horizontal_support(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        return (float(np.clip(point[0], min_x, max_x)), float(start[1]))

    @staticmethod
    def _point_on_horizontal_support(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        min_x = min(start[0], end[0]) - MERGE_EPS
        max_x = max(start[0], end[0]) + MERGE_EPS
        return min_x <= point[0] <= max_x and abs(point[1] - start[1]) <= MERGE_EPS

    def _bar_exists_between(
        self,
        node_u_id: int | None,
        node_v_id: int | None,
        p0: tuple[float, float],
        p1: tuple[float, float],
    ) -> bool:
        if node_u_id is not None and node_v_id is not None:
            for bar in self.bars.values():
                if not bar.active:
                    continue
                if {bar.node_u, bar.node_v} == {node_u_id, node_v_id}:
                    return True
        for bar in self._sorted_active_bars():
            q0, q1 = bar_endpoints(bar, self.nodes)
            if self._points_match(p0, q0) and self._points_match(p1, q1):
                return True
            if self._points_match(p0, q1) and self._points_match(p1, q0):
                return True
        return False

    def _has_exact_overlap(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
    ) -> bool:
        candidate = LineString([p0, p1])
        for bar in self._sorted_active_bars():
            q0, q1 = bar_endpoints(bar, self.nodes)
            overlap = candidate.intersection(LineString([q0, q1]))
            if overlap.geom_type in {"LineString", "MultiLineString"} and overlap.length > MERGE_EPS:
                return True
        return False

    def _segment_hits_cliff_interior(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        anchor_resolution: EndpointResolution,
        second_resolution: EndpointResolution,
    ) -> bool:
        allowed_left = []
        allowed_right = []
        if self._resolution_support_kind(anchor_resolution) == "left_support":
            allowed_left.append(p0)
        if self._resolution_support_kind(second_resolution) == "left_support":
            allowed_left.append(p1)
        if self._resolution_support_kind(anchor_resolution) == "right_support":
            allowed_right.append(p0)
        if self._resolution_support_kind(second_resolution) == "right_support":
            allowed_right.append(p1)
        for frac in np.linspace(0.05, 0.95, 19):
            x = p0[0] + frac * (p1[0] - p0[0])
            y = p0[1] + frac * (p1[1] - p0[1])
            if self._point_in_cliff(x, y, allowed_left, allowed_right):
                return True
        return False

    def _resolution_support_kind(self, resolution: EndpointResolution) -> str | None:
        if resolution.support_kind is not None:
            return resolution.support_kind
        if resolution.kind == "node" and resolution.node_id is not None:
            node = self.nodes.get(resolution.node_id)
            if node is not None and node.kind in {"left_support", "right_support"}:
                return node.kind
        return None

    def _point_in_cliff(
        self,
        x: float,
        y: float,
        allowed_left: list[tuple[float, float]],
        allowed_right: list[tuple[float, float]],
    ) -> bool:
        point = Point(x, y)
        for allowed in allowed_left:
            if math.hypot(x - allowed[0], y - allowed[1]) <= MERGE_EPS:
                return False
        for allowed in allowed_right:
            if math.hypot(x - allowed[0], y - allowed[1]) <= MERGE_EPS:
                return False
        left_corner = self.left_anchor_point()
        if any(math.hypot(allowed[0] - left_corner[0], allowed[1] - left_corner[1]) <= CORNER_EQUIVALENCE_TOL for allowed in allowed_left):
            if math.hypot(x - left_corner[0], y - left_corner[1]) <= CORNER_EQUIVALENCE_TOL:
                return False
        right_corner = self.right_anchor_point()
        if any(math.hypot(allowed[0] - right_corner[0], allowed[1] - right_corner[1]) <= CORNER_EQUIVALENCE_TOL for allowed in allowed_right):
            if math.hypot(x - right_corner[0], y - right_corner[1]) <= CORNER_EQUIVALENCE_TOL:
                return False
        if self.left_cliff.buffer(-1e-9).contains(point):
            return True
        if self.right_cliff.buffer(-1e-9).contains(point):
            return True
        return False

    def _has_spanning_component(self) -> bool:
        adjacency = self._adjacency()
        visited: set[int] = set()
        for node_id, node in self.nodes.items():
            if node_id in visited or node.kind != "left_support":
                continue
            stack = [node_id]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                visited.add(current)
                stack.extend(adjacency.get(current, []))
            if any(self.nodes[nid].kind == "right_support" for nid in component):
                return True
        return False

    def _supports_well_braced(self) -> bool:
        for support_kind in ("left_support", "right_support"):
            support_nodes = [node for node in self.nodes.values() if node.kind == support_kind and node.incident_bar_ids]
            if not support_nodes:
                return False
            support_bar_ids: set[int] = set()
            for node in support_nodes:
                support_bar_ids.update(node.incident_bar_ids)
            if len(support_bar_ids) < 2:
                return False
            for bar_id in support_bar_ids:
                bar = self.bars.get(bar_id)
                if bar is None or not bar.active:
                    continue
                other_node_id = bar.node_v if self.nodes[bar.node_u].kind == support_kind else bar.node_u
                other_node = self.nodes[other_node_id]
                if other_node.kind != "free":
                    continue
                if len(other_node.incident_bar_ids) < 2:
                    return False
        return True

    def _adjacency(self) -> dict[int, set[int]]:
        adjacency: dict[int, set[int]] = {node_id: set() for node_id in self.nodes}
        for bar in self._sorted_active_bars():
            adjacency.setdefault(bar.node_u, set()).add(bar.node_v)
            adjacency.setdefault(bar.node_v, set()).add(bar.node_u)
        return adjacency

    def _sorted_active_bars(self) -> list[Bar]:
        return sorted([bar for bar in self.bars.values() if bar.active], key=lambda bar: bar.placement_order)

    def _add_node(self, x: float, y: float, *, kind: str, movable: bool) -> Node:
        node = Node(self._next_node_id, float(x), float(y), kind=kind, movable=movable)
        self.nodes[node.id] = node
        self._next_node_id += 1
        return node

    def _add_bar(self, node_u: int, node_v: int, *, placement_order: int) -> Bar:
        bar = Bar(self._next_bar_id, node_u, node_v, placement_order=placement_order, active=True)
        self.bars[bar.id] = bar
        self.nodes[node_u].incident_bar_ids.add(bar.id)
        self.nodes[node_v].incident_bar_ids.add(bar.id)
        self._next_bar_id += 1
        return bar

    @staticmethod
    def _midpoint_action_to_endpoints(x: float, y: float, theta: float) -> tuple[tuple[float, float], tuple[float, float]]:
        half = MEMBER_LENGTH / 2.0
        dx = half * math.cos(theta)
        dy = half * math.sin(theta)
        return (x - dx, y - dy), (x + dx, y + dy)

    @staticmethod
    def _parse_anchor_action(
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < 3:
            raise ValueError(f"Expected anchor action with at least 3 values, got shape {action_arr.shape}")
        anchor_x = float(action_arr[0])
        anchor_y = float(action_arr[1])
        theta_anchor = wrap_angle(float(action_arr[2]))
        distance = float(np.clip(action_arr[3], 0.0, MAX_MEMBER_LENGTH)) if action_arr.size >= 4 else float(MEMBER_LENGTH)
        return anchor_x, anchor_y, theta_anchor, distance

    @staticmethod
    def _points_match(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= MERGE_EPS

    def _build_observation(self) -> np.ndarray:
        return self._build_observation_from_bars(self._sorted_active_bars())

    def _build_observation_from_bars(self, bars: list[Bar]) -> np.ndarray:
        obs = np.zeros(1 + 5 * self.max_bars, dtype=np.float32)
        obs[0] = np.float32(self.chasm_width)
        for idx, bar in enumerate(bars[: self.max_bars]):
            anchor_point, second_point = bar_oriented_endpoints(bar, self.nodes)
            base = 1 + 5 * idx
            obs[base : base + 5] = np.array(
                [anchor_point[0], anchor_point[1], second_point[0], second_point[1], 1.0],
                dtype=np.float32,
            )
        return obs

