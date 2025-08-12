import torch
import logging

from FastEHR.dataloader import FoundationalDataModule
from example_wrappers import t2d_inclusion_method


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 5
    print(f"Using device: {device}.")
    print(f"Fitting dataset over {num_threads} threads")

    # Build
    data_dir = "/home/ubuntu/Documents/GitHub/SurvivEHR/FastEHR/examples/data/_built/"
    # overwrite_meta_information:
    #   There is no need to over-write this yet.
    #   In creating the dataset, we collect values which can be used by default,
    #   we can then change these, and pass them into it again to load the dataset.
    dm = FoundationalDataModule(
        path_to_db=data_dir + "example_database.db",
        path_to_ds=data_dir + "indexed_datasets/T2D_hypertension/",
        load=True,
        include_diagnoses=True,
        include_measurements=True,
        drop_missing_data=False,
        drop_empty_dynamic=True,
        tokenizer="tabular",
        overwrite_practice_ids=data_dir + "dataset/practice_id_splits.pickle",
        overwrite_meta_information=data_dir + "dataset/meta_information.pickle",
        study_inclusion_method=t2d_inclusion_method(outcomes=["HYPERTENSION"]),
        num_threads=5,
    )

    vocab_size = dm.train_set.tokenizer.vocab_size

    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")

    for batch in dm.train_dataloader():
        break
    print(batch)

    dm.train_set.view_sample(1)
