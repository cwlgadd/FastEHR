# Expose key modules/classes at the package level
from .dataset.dataset_polars import PolarsDataset
from .tokenizers_local.tokenizers_local import NonTabular, Tabular
from .utils.study_criteria import index_inclusion_method
from .foundational_loader import FoundationalDataModule

# Define __all__ to limit what gets imported with `from FastEHR.dataloader import *`
__all__ = ["PolarsDataset",
           "NonTabular", "Tabular",
           "index_inclusion_method",
           "FoundationalDataModule"]
