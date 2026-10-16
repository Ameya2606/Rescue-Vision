from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import json


@dataclass
class PriorityInfo:
    """Rescue priority level details."""
    level: str         # 'P1', 'P2', 'P3', 'P4'
    name: str          # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    code: str          # 'P1 - CRITICAL', etc.
    emoji: str         # '🔴', '🟠', '🟡', '🟢'
    action: str        # 'Immediate rescue deployment', etc.
    min_confidence: float
    max_confidence: float


# Standard rescue priority definitions
PRIORITY_DEFINITIONS: Dict[str, PriorityInfo] = {
    'P1': PriorityInfo(
        level='P1',
        name='CRITICAL',
        code='P1 - CRITICAL',
        emoji='🔴',
        action='Immediate rescue deployment',
        min_confidence=0.90,
        max_confidence=1.00
    ),
    'P2': PriorityInfo(
        level='P2',
        name='HIGH',
        code='P2 - HIGH',
        emoji='🟠',
        action='Urgent assessment needed',
        min_confidence=0.70,
        max_confidence=0.90
    ),
    'P3': PriorityInfo(
        level='P3',
        name='MEDIUM',
        code='P3 - MEDIUM',
        emoji='🟡',
        action='Schedule inspection',
        min_confidence=0.50,
        max_confidence=0.70
    ),
    'P4': PriorityInfo(
        level='P4',
        name='LOW',
        code='P4 - LOW',
        emoji='🟢',
        action='No immediate action',
        min_confidence=0.00,
        max_confidence=0.50
    )
}


def map_priority(damage_confidence: float) -> PriorityInfo:
    """Map a damage confidence score (0.0 to 1.0) to a rescue priority."""
    if damage_confidence > 0.90:
        return PRIORITY_DEFINITIONS['P1']
    elif damage_confidence > 0.70:
        return PRIORITY_DEFINITIONS['P2']
    elif damage_confidence > 0.50:
        return PRIORITY_DEFINITIONS['P3']
    else:
        return PRIORITY_DEFINITIONS['P4']


@dataclass
class RescuePrediction:
    """Structured result returned by the RescueVision model."""
    prediction: str                      # 'damage' or 'no_damage'
    is_damaged: bool                     # True if prediction == 'damage'
    damage_confidence: float             # Probability between 0.0 and 1.0
    damage_percentage: str               # Formatted string e.g. "94.2%"
    no_damage_confidence: float          # Probability between 0.0 and 1.0
    priority: str                        # e.g. 'P1 - CRITICAL'
    priority_level: str                  # 'P1', 'P2', 'P3', 'P4'
    priority_name: str                   # 'CRITICAL', 'HIGH', etc.
    emoji: str                           # '🔴', '🟠', '🟡', '🟢'
    action: str                          # Recommended rescue operational action
    raw_score: float                     # Direct sigmoid output from model
    image_source: Optional[str] = None   # File path or source identifier if provided
    latitude: Optional[float] = None     # Extracted GPS Latitude if available
    longitude: Optional[float] = None    # Extracted GPS Longitude if available

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to a Python dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert prediction to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        loc_str = f" [Lat: {self.latitude}, Lon: {self.longitude}]" if self.latitude is not None else ""
        return (
            f"{self.emoji} [{self.prediction.upper()}] ({self.damage_percentage} damage confidence) | "
            f"Priority: {self.priority} -> {self.action}{loc_str}"
        )
