"""Dataset classes for multimodal learning."""

from .cremad import CREMADDataset
from .ave import AVEDataset
from .kinetics_sounds import KineticsSoundsDataset

__all__ = ["CREMADDataset", "AVEDataset", "KineticsSoundsDataset"]
