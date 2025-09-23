from typing import Union

import pandas as pd
import numpy as np
import torch


class ConvertToBEHRT(object):
    """
    Convert tokenized FastEHR patient sequences into BEHRT-compatible format.

    This adapter:
    
    - Extends an existing FastEHR tokenizer to include BEHRT's required special tokens.
    - Converts sequences of events grouped by visit into the token/age format
      expected by BEHRT, adding `[CLS]` at the start and `[SEP]` between visits.
    - Retains values (despite not being used in BEHRT).
    - Removes baseline information (e.g. ethnicity, gender) as this is not used by BEHRT.

    Attributes
 
    - **special_tokens** (dict[str, int]): Mapping of BEHRT special tokens to fixed IDs:
      PAD=0, UNK=1, SEP=2, CLS=3, MASK=4.
    - **fastehr_tokenizer** (object): Original FastEHR tokenizer instance passed at init.
    - **supervised** (bool): Whether conversion targets a supervised task (affects final SEP).
    - **tokenizer** (dict[str, int]): Token to index mapping incl. BEHRT specials and original codes.


    Example::

        >>> converter = ConvertToBEHRT(fastehr_tokenizer)
        >>> processed_list_of_patient_dicts = converter(list_of_patient_dicts)
    """
    special_tokens = {"PAD": 0,
                      "UNK": 1,
                      "SEP": 2,
                      "CLS": 3,
                      "MASK": 4
                      }

    def create_behrt_tokenizer(self, tokenizer):

        token2idx = self.special_tokens
        token_counter = 5
        for item, key in tokenizer._stoi.items():
            if item not in self.special_tokens.keys():
                token2idx.update({item: token_counter})
                token_counter += 1

        return token2idx

    def __init__(self, tokenizer, supervised=False):
        self.fastehr_tokenizer = tokenizer
        self.supervised = supervised

        # First use the existing FastEHR tokenizer (with only PAD and
        # UNK as special keys) to a mapping dictionary which will be
        # suitable for the produced BEHRT data
        self.tokenizer = self.create_behrt_tokenizer(tokenizer)

    def __call__(self, data: list[dict]):

        new_data = []
        for datum in data:
            new_data.append(
                self.convert_sample(datum)
            )

        return new_data

    def convert_sample(self, data_sample: dict):

        # Create first temporal value
        tokens = [self.tokenizer["CLS"]]
        ages = [data_sample["ages"][0].item()]
        values = [np.nan]

        for tkn, age, value in zip(
                data_sample["tokens"],
                data_sample["ages"],
                data_sample["values"],
        ):

            # If we are at the end of a visit, then create the SEP token
            next_is_new_visit = False if np.isclose(age, ages[-1]) else True
            if next_is_new_visit:
                tokens.append(self.tokenizer["SEP"])
                ages.append(ages[-1])
                values.append(np.nan)

            # Add next value
            tokens.append(self.tokenizer[self.fastehr_tokenizer.decode([tkn.tolist()])])
            ages.append(age.item())
            values.append(value.item())

        # Add final SEP token to mark end of last visit
        if not self.supervised:
            tokens.append(self.tokenizer["SEP"])
            ages.append(ages[-1])
            values.append(np.nan)
        else:
            if tokens[-2] != self.tokenizer["SEP"]:
                # If supervised and final context token is SEP we dont need to do anything
                # otherwise, we need to add this SEP token so when the target gets stripped
                # later it will end with the correct format.
                # This would only be needed if the target happens instantaneously.
                tokens = tokens[:-1] + [self.tokenizer["SEP"]] + [tokens[-1]]
                ages = ages[:-1] + [ages[-2]] + [ages[-1]]
                values = values[:-1] + [np.nan] + [values[-1]]

        new_datum = {
            "static_covariates": torch.empty(0),
            "tokens":            torch.tensor(tokens),
            "ages":              torch.tensor(ages),
            "values":            torch.tensor(values),
        }

        return new_datum


class BehrtDFBuilder:
    """
    Build a BEHRT-ready DataFrame from batches of token and age tensors.

    Each batch must be shaped [batch_size, seq_len].
    """

    def __init__(
            self,
            token_map:              dict,
            pad_token:              Union[int, str] = "PAD",
            class_token:            Union[int, str] = "CLS",
            sep_token:              Union[int, str] = "SEP",
            id_prefix:              str = "P",
            zfill:                  int = 3,
            min_seq_len:            int = 5,
    ):
        """
        Parameters
        ----------
        token_map : dict
            Mapping from token string to token id (BEHRT-modified vocab).
        pad_token, class_token, sep_token : str or int
            Special tokens (as names or ids)
        id_prefix : str
            Prefix for generated patient IDs.
        zfill : int
            Zero-padding length for patient IDs.
        min_seq_len : int
            Minimum number of non-CLS/SEP tokens required to keep a sample.
            Defaults to 5 as per BEHRT paper.
        """

        # str->id and id->str maps
        self._strtoi = token_map
        self._itostr = {tkn_str: tkn_idx for tkn_idx, tkn_str in token_map.items()}

        # Resolve special token IDs
        self.class_token_id = (
            class_token if isinstance(class_token, int) else self._strtoi[class_token]
        )
        self.pad_token_id = (
            pad_token if isinstance(pad_token, int) else self._strtoi[pad_token]
        )
        self.sep_token_id = (
            sep_token if isinstance(sep_token, int) else self._strtoi[sep_token]
        )

        self.id_prefix = id_prefix
        self.zfill = int(zfill)
        self.min_seq_len = min_seq_len

        self.rows = []
        self.next_id = 1

    def _new_id(self) -> str:
        """Generate a new patient ID."""
        pid = f"{self.id_prefix}{self.next_id:0{self.zfill}d}"
        self.next_id += 1
        return pid

    def _strip_padding(self, tokens_ids, ages):
        """Remove trailing PAD tokens (keeps tokens and ages aligned)."""
        if self.pad_token_id in tokens_ids:
            first_pad = tokens_ids.index(self.pad_token_id)
            tokens_ids = tokens_ids[:first_pad]
            ages = ages[:first_pad]
        return tokens_ids, ages

    def _validate(self, tokens_ids, ages):
        """Run basic validation checks."""
        if len(tokens_ids) != len(ages):
            raise ValueError("Token and age sequences have different lengths.")
        if len(
                [t for t in tokens_ids if t not in (self.class_token_id, self.sep_token_id)]
        ) < self.min_seq_len:
            return False  # too short, skip
        return True

    def add_batch(self, tokens_batch, ages_batch,
                  target_event=None, target_time=None,
                  target_value=None
                  ):
        """
        Add a batch of sequences to the builder.

        :param tokens_batch: Batch of token sequences; each element is a string token
            (or an integer ID).
        :type tokens_batch: torch.Tensor, shape ``[B, T]``
        :param ages_batch: Ages aligned with ``tokens_batch``.
        :type ages_batch: torch.Tensor, shape ``[B, T]``
        :param target_event: Outcome event token/ID for each sequence, or ``None``.
        :type target_event: torch.Tensor or None, shape ``[B]``
        :param target_time: Time-to-event measured from the last token in ``tokens_batch``,
             or ``None``.
        :type target_time: torch.Tensor or None, shape ``[B]``
        :param target_value: Value associated with the outcome event, or ``None``.
        :type target_value: torch.Tensor or None, shape ``[B]``
        
        """        
        # Convert to list of samples
        tokens_batch = tokens_batch.tolist()
        ages_batch = ages_batch.tolist()
        n_samples = len(tokens_batch)
        assert len(ages_batch) == n_samples
        # If optional outcomes are included then do the same
        if target_event is not None and target_time is not None:
            target_event = target_event.tolist()
            target_time = target_time.tolist()
            assert len(target_event) == n_samples
            assert len(target_time) == n_samples
        if target_value is not None:
            target_value = target_value.tolist()
            assert len(target_value) == n_samples

        for sample_idx in range(n_samples):

            token_ids = tokens_batch[sample_idx]
            ages = ages_batch[sample_idx]

            token_ids, ages = self._strip_padding(token_ids, ages)

            if not self._validate(token_ids, ages):
                continue

            # Convert IDs -> strings for storage
            tokens_str = [self._itostr.get(tid, "UNK") for tid in token_ids]

            row_dict = {
                "patid": self._new_id(),
                "caliber_id": tokens_str,
                "age": ages
            }
            if target_event is not None and target_time is not None:
                row_dict.update(
                    {
                        "target_event": self._itostr.get(target_event[sample_idx], "UNK"),
                        "target_time": target_time[sample_idx]
                     }
                )
            if target_value is not None:
                row_dict.update(
                    {
                        "target_value": target_value[sample_idx]
                    }
                )

            # Add sample
            self.rows.append(row_dict)

    def flush(self) -> pd.DataFrame:
        """
        Return a DataFrame of all accumulated rows and clear the buffer.
        This helps manage memory when processing large datasets.
        """
        if not self.rows:
            return pd.DataFrame(columns=["patid", "caliber_id", "age"])
        df = pd.DataFrame(self.rows)
        self.rows.clear()
        return df
