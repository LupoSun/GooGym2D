"""Headless axial truss FEM pipeline for GooGym2D."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from googym2d.config import (
    DEFAULT_AREA,
    DEFAULT_SECOND_MOMENT,
    DEFAULT_UNIT_WEIGHT,
    DEFAULT_YIELD_STRENGTH,
    DEFAULT_YOUNG_MODULUS,
    DEFLECTION_LIMIT,
    LOAD_MAGNITUDE,
)
from googym2d.graph import Bar, FemResult, Node, bar_length

NEAR_MECHANISM_DISPLACEMENT_FACTOR = 1e6


@dataclass(frozen=True)
class SolverElement:
    start_node: int
    end_node: int
    source_bar_id: int
    length: float


@dataclass(frozen=True)
class LoadDistribution:
    nodal_loads: dict[int, tuple[float, float]]
    tributary_lengths: dict[int, float]
    deck_load_total: float
    self_weight_total: float
    span_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrussSolveResult:
    solver_ok: bool
    displacements: np.ndarray
    member_utilization: dict[int, float]
    member_axial_forces: dict[int, float]
    member_failure_mode: dict[int, str]
    support_reactions: dict[int, tuple[float, float]]
    max_displacement: float
    status: str


def _active_nodes_and_bars(
    nodes: dict[int, Node],
    bars: dict[int, Bar],
) -> tuple[dict[int, Node], list[Bar]]:
    active_bars = [bar for bar in bars.values() if bar.active]
    referenced_nodes: set[int] = set()
    for bar in active_bars:
        referenced_nodes.add(bar.node_u)
        referenced_nodes.add(bar.node_v)
    active_nodes = {node_id: nodes[node_id] for node_id in referenced_nodes}
    return active_nodes, sorted(active_bars, key=lambda bar: bar.placement_order)


def _build_solver_elements(
    nodes: dict[int, Node],
    bars: list[Bar],
) -> tuple[dict[int, tuple[float, float]], list[SolverElement]]:
    solver_nodes = {node_id: (float(node.x), float(node.y)) for node_id, node in nodes.items()}
    solver_elements = [
        SolverElement(bar.node_u, bar.node_v, bar.id, max(bar_length(bar, nodes), 1e-9))
        for bar in bars
    ]
    return solver_nodes, solver_elements


def _connected_components(
    nodes: dict[int, Node],
    bars: list[Bar],
) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in nodes}
    for bar in bars:
        adjacency.setdefault(bar.node_u, set()).add(bar.node_v)
        adjacency.setdefault(bar.node_v, set()).add(bar.node_u)

    components: list[set[int]] = []
    remaining = set(nodes)
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _spanning_component_node_ids(
    nodes: dict[int, Node],
    bars: list[Bar],
) -> tuple[int, ...]:
    spanning_components: list[set[int]] = []
    for component in _connected_components(nodes, bars):
        kinds = {nodes[node_id].kind for node_id in component}
        if {"left_support", "right_support"}.issubset(kinds):
            spanning_components.append(component)

    if not spanning_components:
        return tuple(sorted(nodes))

    chosen = max(spanning_components, key=lambda component: (len(component), -min(component)))
    return tuple(sorted(chosen))


def _compute_tributary_lengths(
    node_ids: list[int],
    nodes: dict[int, Node],
    chasm_width: float,
) -> dict[int, float]:
    if not node_ids:
        return {}
    if len(node_ids) == 1:
        return {node_ids[0]: float(chasm_width)}

    xs = [min(max(float(nodes[node_id].x), 0.0), float(chasm_width)) for node_id in node_ids]
    boundaries = [0.0]
    boundaries.extend(0.5 * (xs[idx] + xs[idx + 1]) for idx in range(len(xs) - 1))
    boundaries.append(float(chasm_width))

    tributaries: dict[int, float] = {}
    for idx, node_id in enumerate(node_ids):
        tributaries[node_id] = max(boundaries[idx + 1] - boundaries[idx], 0.0)
    return tributaries


def _assemble_nodal_loads(
    nodes: dict[int, Node],
    bars: list[Bar],
    chasm_width: float,
    *,
    deck_line_load: float,
    area: float,
    unit_weight: float,
) -> LoadDistribution:
    loads = {node_id: np.zeros(2, dtype=np.float64) for node_id in nodes}

    span_node_ids = _spanning_component_node_ids(nodes, bars)
    ordered_span_node_ids = sorted(
        span_node_ids,
        key=lambda node_id: (
            min(max(float(nodes[node_id].x), 0.0), float(chasm_width)),
            float(nodes[node_id].y),
            int(node_id),
        ),
    )
    tributary_lengths = _compute_tributary_lengths(ordered_span_node_ids, nodes, chasm_width)

    deck_load_total = 0.0
    for node_id, tributary in tributary_lengths.items():
        deck_force = float(deck_line_load) * float(tributary)
        loads[node_id][1] -= deck_force
        deck_load_total += deck_force

    self_weight_total = 0.0
    for bar in bars:
        member_weight = float(unit_weight) * float(area) * float(bar_length(bar, nodes))
        self_weight_total += member_weight
        loads[bar.node_u][1] -= 0.5 * member_weight
        loads[bar.node_v][1] -= 0.5 * member_weight

    nodal_loads = {
        node_id: (float(force[0]), float(force[1]))
        for node_id, force in loads.items()
        if not np.allclose(force, 0.0)
    }
    return LoadDistribution(
        nodal_loads=nodal_loads,
        tributary_lengths={int(node_id): float(length) for node_id, length in tributary_lengths.items()},
        deck_load_total=float(deck_load_total),
        self_weight_total=float(self_weight_total),
        span_node_ids=tuple(int(node_id) for node_id in ordered_span_node_ids),
    )


def _assemble_global_stiffness(
    solver_nodes: dict[int, tuple[float, float]],
    elements: list[SolverElement],
    area: float,
    young_modulus: float,
) -> tuple[np.ndarray, dict[int, int]]:
    node_ids = sorted(solver_nodes)
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    dof = 2 * len(node_ids)
    stiffness = np.zeros((dof, dof), dtype=np.float64)

    for element in elements:
        x1, y1 = solver_nodes[element.start_node]
        x2, y2 = solver_nodes[element.end_node]
        length = max(element.length, 1e-9)
        c = (x2 - x1) / length
        s = (y2 - y1) / length
        local = (area * young_modulus / length) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=np.float64,
        )
        dofs = [
            2 * index[element.start_node],
            2 * index[element.start_node] + 1,
            2 * index[element.end_node],
            2 * index[element.end_node] + 1,
        ]
        for i in range(4):
            for j in range(4):
                stiffness[dofs[i], dofs[j]] += local[i, j]
    return stiffness, index


def _element_capacities(
    length: float,
    *,
    area: float,
    second_moment: float,
    young_modulus: float,
    yield_strength: float,
) -> tuple[float, float, float]:
    p_yield = float(area) * float(yield_strength)
    p_euler = (math.pi * math.pi * float(young_modulus) * float(second_moment)) / max(length * length, 1e-9)
    p_capacity = max(min(p_yield, p_euler), 1e-9)
    return p_yield, p_euler, p_capacity


def _support_nodes(nodes: dict[int, Node]) -> dict[int, Node]:
    return {
        node_id: node
        for node_id, node in nodes.items()
        if node.kind in {"left_support", "right_support"}
    }


def _solve_truss(
    solver_nodes: dict[int, tuple[float, float]],
    elements: list[SolverElement],
    support_nodes: dict[int, Node],
    nodal_loads: dict[int, tuple[float, float]],
    *,
    area: float,
    second_moment: float,
    young_modulus: float,
    yield_strength: float,
) -> TrussSolveResult:
    if not nodal_loads or not elements:
        return TrussSolveResult(
            solver_ok=False,
            displacements=np.zeros(0, dtype=np.float64),
            member_utilization={},
            member_axial_forces={},
            member_failure_mode={},
            support_reactions={},
            max_displacement=0.0,
            status="solver_failure",
        )

    stiffness, index = _assemble_global_stiffness(solver_nodes, elements, area, young_modulus)
    dof = stiffness.shape[0]
    forces = np.zeros(dof, dtype=np.float64)
    for node_id, (fx, fy) in nodal_loads.items():
        if node_id not in index:
            continue
        forces[2 * index[node_id]] += float(fx)
        forces[2 * index[node_id] + 1] += float(fy)

    fixed_dofs: list[int] = []
    for node_id, node in support_nodes.items():
        if node_id not in index:
            continue
        node_index = index[node_id]
        if node.kind == "left_support":
            fixed_dofs.extend([2 * node_index, 2 * node_index + 1])
        else:
            fixed_dofs.append(2 * node_index + 1)

    fixed = np.asarray(sorted(set(fixed_dofs)), dtype=np.int64)
    fixed_set = set(int(dof_idx) for dof_idx in fixed.tolist())
    free = np.asarray([dof_idx for dof_idx in range(dof) if dof_idx not in fixed_set], dtype=np.int64)

    if free.size == 0:
        return TrussSolveResult(
            solver_ok=False,
            displacements=np.zeros(dof, dtype=np.float64),
            member_utilization={},
            member_axial_forces={},
            member_failure_mode={},
            support_reactions={},
            max_displacement=0.0,
            status="mechanism",
        )

    reduced = stiffness[np.ix_(free, free)]
    rhs = forces[free]

    try:
        displacements_free = np.linalg.solve(reduced, rhs)
    except np.linalg.LinAlgError:
        return TrussSolveResult(
            solver_ok=False,
            displacements=np.zeros(dof, dtype=np.float64),
            member_utilization={},
            member_axial_forces={},
            member_failure_mode={},
            support_reactions={},
            max_displacement=0.0,
            status="mechanism",
        )

    displacements = np.zeros(dof, dtype=np.float64)
    displacements[free] = displacements_free

    max_displacement = 0.0
    for node_id, node_index in index.items():
        dx = displacements[2 * node_index]
        dy = displacements[2 * node_index + 1]
        max_displacement = max(max_displacement, float(math.hypot(dx, dy)))

    support_reactions: dict[int, tuple[float, float]] = {}
    reactions = stiffness @ displacements - forces
    for node_id in support_nodes:
        if node_id not in index:
            continue
        node_index = index[node_id]
        support_reactions[node_id] = (
            float(reactions[2 * node_index]),
            float(reactions[2 * node_index + 1]),
        )

    member_utilization: dict[int, float] = {}
    member_axial_forces: dict[int, float] = {}
    member_failure_mode: dict[int, str] = {}
    for element in elements:
        x1, y1 = solver_nodes[element.start_node]
        x2, y2 = solver_nodes[element.end_node]
        length = max(element.length, 1e-9)
        c = (x2 - x1) / length
        s = (y2 - y1) / length
        dofs = np.array(
            [
                2 * index[element.start_node],
                2 * index[element.start_node] + 1,
                2 * index[element.end_node],
                2 * index[element.end_node] + 1,
            ],
            dtype=np.int64,
        )
        displacement_vector = displacements[dofs]
        axial_force = (area * young_modulus / length) * np.dot(
            np.array([-c, -s, c, s], dtype=np.float64),
            displacement_vector,
        )
        p_yield, p_euler, p_capacity = _element_capacities(
            length,
            area=area,
            second_moment=second_moment,
            young_modulus=young_modulus,
            yield_strength=yield_strength,
        )
        utilization = abs(float(axial_force)) / p_capacity
        failure_mode = "buckling" if axial_force < 0.0 and p_euler <= p_yield else "yield"
        if utilization >= member_utilization.get(element.source_bar_id, -math.inf):
            member_utilization[element.source_bar_id] = float(utilization)
            member_axial_forces[element.source_bar_id] = float(axial_force)
            member_failure_mode[element.source_bar_id] = failure_mode

    return TrussSolveResult(
        solver_ok=True,
        displacements=displacements,
        member_utilization=member_utilization,
        member_axial_forces=member_axial_forces,
        member_failure_mode=member_failure_mode,
        support_reactions=support_reactions,
        max_displacement=float(max_displacement),
        status="ok",
    )


def run_fea_pipeline(
    nodes: dict[int, Node],
    bars: dict[int, Bar],
    chasm_width: float,
    *,
    load_magnitude: float = LOAD_MAGNITUDE,
    area: float = DEFAULT_AREA,
    second_moment: float = DEFAULT_SECOND_MOMENT,
    young_modulus: float = DEFAULT_YOUNG_MODULUS,
    yield_strength: float = DEFAULT_YIELD_STRENGTH,
    unit_weight: float = DEFAULT_UNIT_WEIGHT,
) -> FemResult:
    """Run terminal structural analysis on the active graph."""

    active_nodes, active_bars = _active_nodes_and_bars(nodes, bars)
    if not active_bars:
        return FemResult(
            solver_ok=False,
            max_utilization=float("inf"),
            member_utilization={},
            max_displacement=float("inf"),
            failed_bar_ids=[],
            status="solver_failure",
        )

    support_nodes = _support_nodes(active_nodes)
    has_left = any(node.kind == "left_support" for node in support_nodes.values())
    has_right = any(node.kind == "right_support" for node in support_nodes.values())
    if not has_left or not has_right:
        return FemResult(
            solver_ok=False,
            max_utilization=float("inf"),
            member_utilization={},
            max_displacement=float("inf"),
            failed_bar_ids=[],
            status="solver_failure",
        )

    solver_nodes, solver_elements = _build_solver_elements(active_nodes, active_bars)
    load_distribution = _assemble_nodal_loads(
        active_nodes,
        active_bars,
        chasm_width,
        deck_line_load=load_magnitude,
        area=area,
        unit_weight=unit_weight,
    )
    solve_result = _solve_truss(
        solver_nodes,
        solver_elements,
        support_nodes,
        load_distribution.nodal_loads,
        area=area,
        second_moment=second_moment,
        young_modulus=young_modulus,
        yield_strength=yield_strength,
    )

    max_utilization = max(solve_result.member_utilization.values(), default=float("inf"))
    failed_bar_ids = sorted([bar_id for bar_id, util in solve_result.member_utilization.items() if util > 1.0])

    if not solve_result.solver_ok:
        return FemResult(
            solver_ok=False,
            max_utilization=float("inf"),
            member_utilization=solve_result.member_utilization,
            max_displacement=float("inf"),
            failed_bar_ids=failed_bar_ids,
            status=solve_result.status,
        )

    status = "ok"
    if solve_result.max_displacement > DEFLECTION_LIMIT * NEAR_MECHANISM_DISPLACEMENT_FACTOR:
        status = "near_mechanism"
    elif max_utilization > 1.0 and failed_bar_ids:
        if any(solve_result.member_failure_mode.get(bar_id) == "buckling" for bar_id in failed_bar_ids):
            status = "buckling"
        else:
            worst_bar_id = max(
                failed_bar_ids,
                key=lambda bar_id: solve_result.member_utilization.get(bar_id, -math.inf),
            )
            status = solve_result.member_failure_mode.get(worst_bar_id, "yield")

    return FemResult(
        solver_ok=True,
        max_utilization=float(max_utilization),
        member_utilization={int(bar_id): float(util) for bar_id, util in solve_result.member_utilization.items()},
        max_displacement=float(solve_result.max_displacement),
        failed_bar_ids=failed_bar_ids,
        status=status,
    )

