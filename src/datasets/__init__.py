"""Dataset classes for multimodal learning."""

from .cremad import CREMADDataset
from .ave import AVEDataset
from .kinetics_sounds import KineticsSoundsDataset
from .mosei import MOSEIDataset
from .cmu_mosi import CMUMOSIDataset
from .sarcasm import SarcasmDataset
from .twitter import TwitterDataset

__all__ = [
    "CREMADDataset", "AVEDataset", "KineticsSoundsDataset",
    "MOSEIDataset", "CMUMOSIDataset",
    "SarcasmDataset", "TwitterDataset",
]
