"""Public package exports for GooGym2D."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from googym2d.config import (
    ALLOWED_CHASM_WIDTHS,
    CLIFF_HEIGHT,
    CLIFF_WIDTH,
    CONTINUOUS_CHASM_WIDTH_RANGE,
    DEFAULT_AREA,
    DEFAULT_SECOND_MOMENT,
    DEFAULT_UNIT_WEIGHT,
    DEFAULT_YIELD_STRENGTH,
    DEFAULT_YOUNG_MODULUS,
    DEFLECTION_LIMIT,
    LOAD_MAGNITUDE,
    MAX_BARS,
    MAX_MEMBER_LENGTH,
    MEMBER_LENGTH,
    MIN_MEMBER_LENGTH,
)
from googym2d.env import BridgeBuildEnv, EndpointResolution, PlacementPreview, wrap_angle
from googym2d.fem import run_fea_pipeline
from googym2d.graph import Bar, FemResult, Node, snapshot_graph

ENV_ID = "GooGym2D-BridgeBuild-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="googym2d.env:BridgeBuildEnv")

__all__ = [
    "ALLOWED_CHASM_WIDTHS",
    "Bar",
    "BridgeBuildEnv",
    "CLIFF_HEIGHT",
    "CLIFF_WIDTH",
    "CONTINUOUS_CHASM_WIDTH_RANGE",
    "DEFAULT_AREA",
    "DEFAULT_SECOND_MOMENT",
    "DEFAULT_UNIT_WEIGHT",
    "DEFAULT_YIELD_STRENGTH",
    "DEFAULT_YOUNG_MODULUS",
    "DEFLECTION_LIMIT",
    "ENV_ID",
    "EndpointResolution",
    "FemResult",
    "LOAD_MAGNITUDE",
    "MAX_BARS",
    "MAX_MEMBER_LENGTH",
    "MEMBER_LENGTH",
    "MIN_MEMBER_LENGTH",
    "Node",
    "PlacementPreview",
    "run_fea_pipeline",
    "snapshot_graph",
    "wrap_angle",
]

