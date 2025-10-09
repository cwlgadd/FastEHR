# Output adapters introduction

The ``Adapter`` wrapper class converts the output of the native FastEHR dataloader to a format applicable for 
different downstream models.

The dataloader can be modified using a simple keyword argument. This can be done either at creation, or on
existing datasets. 

```python Adapted dataloader

dm = FoundationalDataModule(
    path_to_db = "path/to/database.db",
    path_to_ds = "path/to/dataset/",
    adapter = "BEHRT",
)
```

This will then initialise an ``Adapter`` instance inside the dataloader, which is passed to the collator.
Samples are then adapted on-the-fly inside the collator. The ``Adapter`` updates the native FastEHR tokenizer
with a new tokenizer that is applicable to the downstream model. For example, where special tokens are introduced.

For some models a pandas DataFrame is required. Whilst this has obvious flaws, for example requiring all smaples 
to be held in memory at once, this is expected downstream behaviour. In these cases, we additionally supply a
DataFrame builder to iterate over the DataLoader batches and collate them into the appropriately formatted DataFrame.

# Workflows

A workflow for each adapter is shown below.

## BEHRT

This adapter converts native FastEHR datasets and dataloaders to the form expected in [BEHRT](https://github.com/deepmedicine/BEHRT).

After initial creation of an adapted native dataset instance, as shown in the adapter introduction, save the tokenizer
in a form BEHRT will later recognise. Note, though not required we specify a different path for the BEHRT dataset.

```python Convert tokenizer (add special tokens such as 'SEP')
bert_vocab = {'token2idx': dm.adapter.tokenizer}
with open("path/to/BEHRT/dataset/token2idx.pkl", "wb") as f:
    pkl.dump(bert_vocab, f)
```

Next, we convert the DataLoader ``dm``, which currently provides batches in the correct format, to the DataFrame expected in 
BEHRT.

```python
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

    # Add batch to chunks
    builder.add_batch(batch["tokens"], batch["ages"])

    # Update every 10 batches to avoid iterating on DataFrame too heavily
    if idx_batch % 10 == 0:
        df_chunk = builder.flush()
        if not df_chunk.empty:
            chunks.append(df_chunk)

# Final flush
final_chunk = builder.flush()
if not final_chunk.empty:
    chunks.append(final_chunk)

# Concatenate all chunks
df = pd.concat(chunks, ignore_index=True)
df.to_parquet("path/to/BEHRT/dataset/train_dataset.parquet", index=False)
```

This can be repeated for the validation and testing splits.

Within [BEHRT](https://github.com/deepmedicine/BEHRT) we can now use this dataset using the standard API

```python Using Adapted FastEHR dataset in BEHRT
# Load in the tokenizer
BertVocab = load_obj("path/to/BEHRT/dataset/token2idx.pkl")

# Load in the dataset
data = pd.read_parquet("path/to/BEHRT/dataset/train_dataset.parquet")

# and proceed as normal
data['length'] = data['caliber_id'].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
data = data[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)
```

