# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import itertools
import pathlib
import sqlite3
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pickle
import numpy as np
from FastEHR.dataloader.dataset.collector import SQLiteDataCollector
from sklearn.model_selection import train_test_split as sk_split
import logging
from tqdm import tqdm
import psutil
import os
from joblib import Parallel, delayed

class PolarsDataset:
    """

    """

    def __init__(self, path_to_db):
        """Initialize the dataset and establish a connection to the database."""
        super().__init__()
        self.save_path = None
        self.path_to_db = path_to_db
        self.collector = SQLiteDataCollector(self.path_to_db)
        self.collector.connect()           # Persistent connection


    def fit(self,
            path:                                str,
            practice_inclusion_conditions:       Optional[list[str]] = None,
            include_static:                      bool = True,
            include_diagnoses:                   bool = True,
            include_measurements:                bool = True,
            overwrite_practice_ids:              Optional[tuple] = None,    
            overwrite_meta_information:          Optional[str] = None,
            num_threads:                         int = 1,
            **kwargs
           ):
        r"""
        Create a Deep Learning-friendly dataset by extracting structured data from SQLite.

        Load information from SQL tables into polars frames for each table in chunks. For each chunk
           * Then combine, align frames, and put into a DL friendly lazy Polars representation
           * iteratively find normalisation statistics, counts, or any other meta information 
           * Save polars frames to parquets
           * Create a hashmap dictionary which allows us to do faster lookups than native PyArrow solutions


        - Loads SQLite data into Polars frames.
        - Splits data into training, validation, and test sets.
        - Saves preprocessed data in Parquet format.
           
        ARGS:
            path:  
                Full path to folder where parquet files containing the Polars dataset, meta information, and file-look up pickles
            
        KWARGS:
            practice_inclusion_conditions:
                The set of practice inclusion conditions to query against the collector. For example, only include patients from practices where ["COUNTRY = 'E'"]
            include_static:
                Whether to include static information in the meta_information
            include_diagnoses:
                Whether to include diagnoses in the meta_information, and in the parquet dataset
            include_measurements
                Whether to include measurements in the meta_information, and in the parquet dataset
            overwrite_practice_ids:
                If you want to overwrite the practice ID allocations to train/test/validation splits, for example if you are building a fine-tuning dataset
                from within the foundation model dataset you will need to ensure information is not leaked into the test/validation from the pre-trained model's
                training set.
            overwrite_meta_information:
                If you want to overwrite the meta_information, for example using quantile bounds for some measurements, then there is no need
                to pre-process it again. In this case, pass in the path to an existing meta_information pickled file. 

        **KWARGS:
            drop_empty_dynamic: bool = True,

            drop_missing_data: bool = True,
            
            exclude_pre_index_age: bool = False,

        TODO: pickle this class rather than separately saving the different attributes. However the SQLite collector cannot be pickled
            
        """
       
        self.save_path = path
        logging.info(f"Building Polars datasets and saving to {path}")

        try:
            # Train, test, validation split
            if overwrite_practice_ids is None:
                self.train_practice_ids, self.val_practice_ids, self.test_practice_ids = self.__train_test_val_split(
                    practice_inclusion_conditions=practice_inclusion_conditions
                )
                splits = {"train": self.train_practice_ids, "val": self.val_practice_ids, "test": self.test_practice_ids}
                with open(self.save_path + f'practice_id_splits.pickle', 'wb') as handle:
                    pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # If fine-tuning then we want to use existing splits to avoid data leakage
                with open(overwrite_practice_ids, 'rb') as f:
                    splits = pickle.load(f)
                    self.train_practice_ids = splits["train"]
                    self.val_practice_ids = splits["val"]
                    self.test_practice_ids = splits["test"]
                    logging.info(f"Using train/test/val splits from {overwrite_practice_ids}")

            # Collect meta information (single-threaded).
            #    These are pre-calculations, used for torch loader length, tokenization, and standardisation
            if overwrite_meta_information is None:
                # all_train = list(itertools.chain.from_iterable(self.train_practice_ids))
                meta_information = self.collector.get_meta_information(
                    practice_ids=None, static=include_static,
                    diagnoses=include_diagnoses, measurement=include_measurements
                )
                logging.debug(f"meta_information {meta_information}")
                with open(self.save_path + f'meta_information.pickle', 'wb') as handle:
                    pickle.dump(meta_information, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Process splits in parallel, chunking by practice
            #   Looping over practice IDs
            #    * create the generator that performs a lookup on the database for each practice ID and returns a lazy frame for each table's rows
            #    * collating the lazy tables into a lazy DL-friendly representation,
            for split_name, split_ids in zip(["test", "train", "val"],
                                             [self.test_practice_ids, self.train_practice_ids, self.val_practice_ids]):

                logging.info(f"Processing {split_name} split...")

                # Dataset path
                path = pathlib.Path(self.save_path + f"split={split_name}")    # /{chunk_name}.parquet
                assert not any(path.iterdir()), f"Expected empty directory {path}, but found {[_ for _ in path.iterdir()]}"
                # TODO: make directories if they dont already exist

                # Create parquet train/test/val dataset
                self.__save_data_to_parquet(
                    save_path=path, split_ids=split_ids, include_diagnoses=include_diagnoses,
                    include_measurements =include_measurements, num_threads=num_threads, **kwargs
                )

                # Index the dataset for faster lookups
                logging.debug(f"Creating file_row_count_dicts for file-index look-ups")
                hashmap = self.__index_parquet_file_rows(path)
                with open(self.save_path + f'file_row_count_dict_{split_name}.pickle', 'wb') as handle:
                    pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

        finally:
            self.collector.disconnect()  # Ensure database is closed properly


    def __train_test_val_split(self, practice_inclusion_conditions=None):
        """
        Splits the dataset into training, validation, and test sets based on unique practice IDs.

        This method retrieves a list of distinct `PRACTICE_ID`s from the database, optionally
        filtering them based on the provided inclusion conditions. It then performs a
        stratified split into training (90%), validation (5%), and test (5%) subsets.

        Args:
            practice_inclusion_conditions (Optional[list[str]]):
                A list of SQL conditions to filter practices before splitting.
                Example: ["COUNTRY = 'E'"] to include only practices from England.

        Returns:
            tuple:
                - train_practice_ids (list[str]): Practice IDs allocated for training.
                - val_practice_ids (list[str]): Practice IDs allocated for validation.
                - test_practice_ids (list[str]): Practice IDs allocated for testing.

        Notes:
            - If `practice_inclusion_conditions` is `None`, all practice IDs are used.
            - Uses `sklearn.model_selection.train_test_split` to randomly partition the data.
              further split into equal validation and test sets (5% each).
        """

        # get a list of practice IDs which are used to chunk the database
        #    We can optionally subset which practice IDs should be included in the study based on some criteria.
        #    For example, asserting that we only want practices in England can be achieved by adding an inclusion
        #    that applies to the static table
        logging.info(f"Chunking by unique practice ID with {'no' if practice_inclusion_conditions is None else practice_inclusion_conditions} practice inclusion conditions")
        practice_ids = self.collector._extract_distinct(table_names=["static_table"],
                                                        identifier_column="PRACTICE_ID",
                                                        inclusion_conditions=practice_inclusion_conditions
                                                       )
        
        # Create train/test/val splits, each is list of practice_id
        logging.info(f"Creating train/test/val splits using practice_ids")
        train_practice_ids, test_practice_ids = sk_split(practice_ids, test_size=0.1)
        test_practice_ids, val_practice_ids = sk_split(test_practice_ids, test_size=0.5)

        return train_practice_ids, val_practice_ids, test_practice_ids

    def __save_data_to_parquet(self,
                               save_path: str,
                               split_ids: list[str],
                               include_diagnoses: bool = True,
                               include_measurements: bool = True,
                               num_threads: int = 1,
                               **kwargs,
                               ) -> pl.LazyFrame:
        r"""
        Saves extracted practice data as a Polars dataset in Parquet format.

        Each thread processes one practice ID at a time using its own database connection
        and generator for thread safety.

        Writes the dataset to Parquet using a persistent connection.
            - Uses a **separate connection per thread** for multi-threading.
            - **Reuses the persistent connection** when running in a single-threaded mode.

        ARGS:

        KWARGS:

        """
        logging.debug(f"Generating dataset over practice IDs")

        # Single-threaded execution: Use the persistent SQLite connection
        if num_threads == 1:

            # Using persistent collector, loop over all split_ids and create parquet dataset
            total_samples = self.__save_split_data_to_parquet(
                save_path, split_ids, include_diagnoses, include_measurements, self.collector, **kwargs
            )

        # Multi-threaded execution: Create a separate SQLite connection per thread
        elif num_threads > 1:

            # **Split the workload**: Assign each thread a subset of practice IDs
            split_batches = np.array_split(split_ids, num_threads)

            # Using thread safe collectors, in each thread loop over the sub-batch of split_ids and create parquet
            Parallel(n_jobs=num_threads, prefer="threads", verbose=10)(
                delayed(self.__save_split_data_to_parquet)(
                    save_path, split_ids_thread, include_diagnoses, include_measurements,
                    SQLiteDataCollector(self.path_to_db), **kwargs
                )
                for split_ids_thread in split_batches)

            total_samples = f"unknown (multi-threaded processing)"

        else:
            raise NotImplementedError

        logging.info(f"Created dataset at {save_path} with {total_samples} number of samples")

        return

    def __save_split_data_to_parquet(self,
                                     save_path:               str,
                                     split_ids:               list[str],
                                     include_diagnoses:       bool,
                                     include_measurements:    bool,
                                     collector,
                                     **kwargs,
                                    ):
        """
        Writes a chunk of data to Parquet using a **thread-safe SQLite connection**.
            save splits with a hive partitioning

        Args:
            lazy_table_frames_dict (dict): Dictionary of lazy tables for a practice.
            chunk_name (str): The identifier for the current chunk.
            save_path (str): Path where the parquet files will be stored.
            collector (SQLiteDataCollector): The **SQLite connection assigned to this thread**.
        """
        # Create the generator, which returns the table contents of qualifying practices one at a time
        # Can process entire list of IDs at once by changing to `distinct_values=[split_ids]`
        practice_generator = collector._generate_lazy_by_distinct(distinct_values=split_ids,
                                                                  identifier_column="PRACTICE_ID",
                                                                  include_diagnoses=include_diagnoses,
                                                                  include_measurements=include_measurements,
                                                                  )

        total_samples = 0
        for chunk_name, lazy_table_frames_dict in tqdm(practice_generator, total=len(split_ids)):


            logging.debug(f"Processing {chunk_name}")

            # Ensure the collector's connection is open (for safety in multi-threading)
            collector.connect()  # If already connected (i.e. persistent connection), this is a no-op

            try:
                # Merge the lazy polars tables provided by the generator into one lazy polars frame
                lazy_batch = collector._collate_lazy_tables(lazy_table_frames_dict, **kwargs)

                # Collect the lazy Polars frame, materializing the batch
                #   include row count so we can filter when reading from file
                df = lazy_batch.collect().with_row_count().to_pandas()  # offset=total_samples
                total_samples = len(df.index)

                if total_samples > 0:
                    # Convert row count to lower cardinality bins for efficient partitioning
                    # ... the smaller the window the more files created + storage space used, and the longer this takes to run
                    # ... but the faster the read efficiency.
                    df = df.assign(CHUNK = [int(_v / 250) for _v in df['row_nr']])

                    # Convert to an Apache Arrow Table and save to a partitioned Parquet dataset
                    table = pa.Table.from_pandas(df)
                    pq.write_to_dataset(table, root_path=save_path, partition_cols=['COUNTRY', 'HEALTH_AUTH', 'PRACTICE_ID', 'CHUNK'])

                    logging.debug(f'Saved {total_samples} rows for chunk {chunk_name}')

            except Exception as e:
                logging.exception(f"Error processing {chunk_name}: {e}")

            finally:
                if isinstance(collector, SQLiteDataCollector) and collector is not self.collector:
                    collector.disconnect()  # Close connection after processing if multi-threaded mode

                logging.debug(f"Finished processing {chunk_name}")

        return total_samples

    def __index_parquet_file_rows(self, parquet_path):
        # Get all files at specified path, and extract from meta data how many samples are in each file. This allows for for faster reading during calls to dataset
        file_row_counts = {}
        desc = "Getting file row counts. This allows the creation of an index to file map, increasing read efficiency"
        total_count = 0
        for file in tqdm(pathlib.Path(parquet_path).rglob('*.parquet'), desc=desc):
            num_rows =  pq.ParquetFile(file).metadata.num_rows
            relative_file_path = str(file)[len(str(parquet_path)):]               # remove the root of the file path
            logging.debug(f"relative_file_path: {relative_file_path}  -  manually done last dataset build, verify this is correct on next run and delete.")
            file_row_counts[file] = num_rows                                      # update hash map
            total_count += num_rows

        logging.info(f"\t Obtained with a total of {total_count} samples")
    
        return file_row_counts