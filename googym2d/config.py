"""Shared constants for the GooGym2D bridge environment."""

from __future__ import annotations

import math

MAX_BARS: int = 20

MEMBER_LENGTH: float = 3.0
MIN_MEMBER_LENGTH: float = 0.5 * MEMBER_LENGTH
MAX_MEMBER_LENGTH: float = 1.25 * MEMBER_LENGTH

CLIFF_HEIGHT: float = 10.0
CLIFF_WIDTH: float = 5.0
ALLOWED_CHASM_WIDTHS: tuple[float, ...] = (10.0,)
CONTINUOUS_CHASM_WIDTH_RANGE: tuple[float, float] = (
    float(min(ALLOWED_CHASM_WIDTHS)),
    float(max(ALLOWED_CHASM_WIDTHS)),
)
DEFAULT_CHASM_WIDTH_SAMPLING_MODE: str = "discrete"

NODE_SNAP_RADIUS: float = 1.0
SUPPORT_SNAP_RADIUS: float = 1.0
SECONDARY_DIRECTION_SNAP_RAD: float = math.radians(15.0)
MERGE_EPS: float = 1e-6
CORNER_EQUIVALENCE_TOL: float = 0.5
INVALID_ACTION_PENALTY: float = -0.1
DEFLECTION_LIMIT: float = 0.5 * MEMBER_LENGTH
TRAIN_START_ZONE_WIDTH: float = MEMBER_LENGTH
TRAIN_GOAL_ZONE_WIDTH: float = CLIFF_WIDTH

# SI-unit axial truss interpretation for a 10 m pedestrian bridge benchmark.
DEFAULT_YOUNG_MODULUS: float = 210e9
DEFAULT_DENSITY: float = 7850.0
DEFAULT_UNIT_WEIGHT: float = 78.5e3
DEFAULT_YIELD_STRENGTH: float = 355e6
DEFAULT_AREA: float = 1.8e-3
DEFAULT_SECOND_MOMENT: float = 1.18e-6
DECK_DEAD_LOAD_PER_M: float = 1.0e3
DECK_LIVE_LOAD_PER_M: float = 4.0e3
LOAD_MAGNITUDE: float = DECK_DEAD_LOAD_PER_M + DECK_LIVE_LOAD_PER_M

