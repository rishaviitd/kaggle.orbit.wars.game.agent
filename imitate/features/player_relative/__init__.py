"""Player-canonical orientation and opponent-slot feature extraction."""

from .extractor import (
    FLEET_PLAYER_RELATIVE_FEATURES,
    PLANET_PLAYER_RELATIVE_FEATURES,
    extract_player_relative_features,
)

__all__ = [
    "FLEET_PLAYER_RELATIVE_FEATURES",
    "PLANET_PLAYER_RELATIVE_FEATURES",
    "extract_player_relative_features",
]
