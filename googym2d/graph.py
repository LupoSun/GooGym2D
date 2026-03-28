"""Graph types and geometry helpers for GooGym2D."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

NodeKind = Literal["left_support", "right_support", "free"]


@dataclass
class Node:
    """Graph vertex used by the bridge editor and FEM solver."""

    id: int
    x: float
    y: float
    kind: NodeKind
    movable: bool
    incident_bar_ids: set[int] = field(default_factory=set)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "kind": self.kind,
            "movable": bool(self.movable),
            "incident_bar_ids": sorted(self.incident_bar_ids),
        }


@dataclass
class Bar:
    """Graph edge placed during the episode."""

    id: int
    node_u: int
    node_v: int
    placement_order: int
    active: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "node_u": self.node_u,
            "node_v": self.node_v,
            "placement_order": self.placement_order,
            "active": bool(self.active),
        }


@dataclass
class FemResult:
    """Terminal structural analysis summary."""

    solver_ok: bool
    max_utilization: float
    member_utilization: dict[int, float]
    max_displacement: float
    failed_bar_ids: list[int]
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "solver_ok": bool(self.solver_ok),
            "max_utilization": float(self.max_utilization),
            "member_utilization": {
                int(bar_id): float(util)
                for bar_id, util in self.member_utilization.items()
            },
            "max_displacement": float(self.max_displacement),
            "failed_bar_ids": [int(bar_id) for bar_id in self.failed_bar_ids],
            "status": self.status,
        }


def bar_endpoints(bar: Bar, nodes: dict[int, Node]) -> tuple[tuple[float, float], tuple[float, float]]:
    node_u = nodes[bar.node_u]
    node_v = nodes[bar.node_v]
    return (float(node_u.x), float(node_u.y)), (float(node_v.x), float(node_v.y))


def bar_oriented_endpoints(bar: Bar, nodes: dict[int, Node]) -> tuple[tuple[float, float], tuple[float, float]]:
    return bar_endpoints(bar, nodes)


def bar_length(bar: Bar, nodes: dict[int, Node]) -> float:
    (x1, y1), (x2, y2) = bar_endpoints(bar, nodes)
    return math.hypot(x2 - x1, y2 - y1)


def bar_midpoint(bar: Bar, nodes: dict[int, Node]) -> tuple[float, float]:
    (x1, y1), (x2, y2) = bar_endpoints(bar, nodes)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bar_theta(bar: Bar, nodes: dict[int, Node]) -> float:
    (x1, y1), (x2, y2) = bar_endpoints(bar, nodes)
    return math.atan2(y2 - y1, x2 - x1)


def bar_geometry(bar: Bar, nodes: dict[int, Node]) -> tuple[float, float, float, float]:
    mid_x, mid_y = bar_midpoint(bar, nodes)
    theta = bar_theta(bar, nodes)
    length = bar_length(bar, nodes)
    return mid_x, mid_y, theta, length


def snapshot_graph(nodes: dict[int, Node], bars: dict[int, Bar]) -> dict[str, object]:
    active_nodes = {node_id for node_id, node in nodes.items() if node.incident_bar_ids or node.kind != "free"}
    return {
        "nodes": [nodes[node_id].as_dict() for node_id in sorted(active_nodes)],
        "bars": [bars[bar_id].as_dict() for bar_id in sorted(bars)],
    }

