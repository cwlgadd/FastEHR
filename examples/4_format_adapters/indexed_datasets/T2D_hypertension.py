import torch
import logging
import pickle as pkl
import pandas as pd

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.adapters.BEHRT import BehrtDFBuilder


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 5
    print(f"Using device: {device}.")
    print(f"Fitting dataset over {num_threads} threads")

    # Load with adapter
    # Note: you may face problems when loading an existing dataset from a new directory,
    #       this is because the hash map used for row look-ups stores the relative import
    #       from the original dataset.
    data_dir = "/home/ubuntu/Documents/GitHub/SurvivEHR/FastEHR/examples/data/_built/"
    dm = FoundationalDataModule(
        path_to_db=data_dir + "example_database.db",
        path_to_ds=data_dir + "indexed_datasets/T2D_hypertension/",
        overwrite_practice_ids=data_dir + "dataset/practice_id_splits.pickle",
        overwrite_meta_information=data_dir + "dataset/meta_information.pickle",
        load=True,
        adapter="BEHRT",
        supervised=True,
    )

    vocab_size = dm.train_set.tokenizer.vocab_size

    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")

    # Save built tokenizer to file
    bert_vocab = {'token2idx': dm.adapter.tokenizer}
    with open(data_dir + "adapted/BEHRT/T2D_hypertension/token2idx.pkl", "wb") as f:
        pkl.dump(bert_vocab, f)

    builder = BehrtDFBuilder(
        token_map=dm.adapter.tokenizer,
        pad_token="PAD",
        class_token="CLS",
        sep_token="SEP",
        id_prefix="P",
        zfill=7,
        min_seq_len=2,
    )

    chunks = []
    for idx_batch, batch in enumerate(dm.train_dataloader()):

        builder.add_batch(batch["tokens"], batch["ages"],
                          target_event=batch["target_token"],
                          target_time=batch["target_age_delta"],
                          target_value=batch["target_value"],
                          )

        if idx_batch % 10 == 0:
            df_chunk = builder.flush()

            if not df_chunk.empty:
                chunks.append(df_chunk)

    # Final flush
    final_chunk = builder.flush()
    if not final_chunk.empty:
        chunks.append(final_chunk)

    # Concatenate all chunks (or return empty df)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(data_dir + "adapted/BEHRT/T2D_hypertension/dataset.parquet", index=False)

        print(len(df))
        print(df.head())
        # print(df["patid"][0])
        # print(df["caliber_id"][0])
        # print(df["age"][0])
    else:
        logging.warning("No valid data")
