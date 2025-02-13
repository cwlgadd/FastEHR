import os
from pathlib import Path
import sys
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from FastEHR.dataloader.foundational_loader import FoundationalDataModule
import logging
import time


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 5
    print(f"Using device: {device}.")
    print(f"Fitting dataset over {num_threads} threads")

    # Build
    # overwrite_meta_information:
    #   There is no need to over-write this yet.
    #   In creating the dataset, we collect values which can be used by default, we can then change these, and pass them into it again to load the dataset.
    dm = FoundationalDataModule(path_to_db="../data/_built/example_database.db",
                                path_to_ds="../data/_built/dataset/",
                                load=True,
                                include_diagnoses=True,                            
                                include_measurements=True,
                                drop_missing_data=False,
                                drop_empty_dynamic=True,
                                tokenizer="tabular",
                                practice_inclusion_conditions=["COUNTRY = 'E'"],
                                overwrite_meta_information=None,         
                                num_threads=num_threads,
                                supervised=True,
                                supervised_time_scale=1
                               )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")

    print(dm.train_set.view_sample(1))

    for batch in dm.train_dataloader():
        break
    print(batch)
    
    