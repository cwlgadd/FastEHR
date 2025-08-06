import polars as pl


class index_inclusion_method():
    """
    Filters a dataset based on indexing criteria, outcomes, and study constraints.

    Parameters
    ----------
    index_on (Union[str, List[str], float, int]):
        Defines the indexing criteria:
        - If indexing on an event, provide the event name as a string.
        - If indexing on multiple events, provide a list of event name strings.
        - If indexing by age, provide a float or integer representing age in days.
        - When multiple index events exist, the first valid index date is taken.

    outcomes (Union[List[str], Callable]):
        Specifies the outcomes of interest:
        - A list of event names to be used as outcomes.
        - A callable object that filters the event column of the dataframe for the desired outcomes.

    require_outcome (bool, optional):
        Whether the outcome must be observed within the study period constraints.
        - If `False`, includes patients who have not yet seen the outcome (e.g., survival analysis).
        - If `True`, includes only patients who have observed the outcome, though its value may still be missing.

    include_on_events_prior_to_index (Tuple[str, int], optional):
        Filters patients based on prior events before the index date.
        - The first element is a string indicating the event token.
        - The second element is an integer representing the number of days before the index event.
        - Example: If studying medication effects post-diagnosis, you may include only those diagnosed 60 days before medication.

    exclude_on_events_prior_to_index (List[str], optional):
        Excludes patients based on events occurring before the index date.
        - Example 1: If studying the initiation of a medication, patients already on the medication may be excluded.

    exclude_on_events (List[str], optional):
        Excludes patients based on whether they have experienced an event at any time.
        - Example: If studying Type II diabetes, patients with a Type I diabetes diagnosis may be excluded.

    study_period (List[str], optional):
        Defines the study period in the format `["yyyy-mm-dd", "yyyy-mm-dd"]` in chronological order.
        - The start of the study period does not determine the start of observations but contributes to defining the indexing period.
        - The study end date marks the end of observations.

    age_at_entry_range (List[int], optional):
        Defines the allowable age range for cohort entry in years `[min_age, max_age]`.

    min_registered_years (int, optional):
        The minimum number of years a patient must be registered at the practice for inclusion at cohort entry.

    min_events (int, optional):
        The minimum number of events a patient must have experienced (up to and including the index event) to be included in the study.

    Notes:
    - This function is executed on a per-practice basis, so there is no concern about overlapping `PATIENT_ID` values.

    """
    def __init__(self,
                 index_on,
                 outcomes,
                 require_outcome=False,
                 include_on_events_prior_to_index=None,
                 exclude_on_events_prior_to_index=None,
                 exclude_on_events=None,
                 study_period=["1998-01-01", "2019-12-31"],
                 age_at_entry_range=[25, 85],
                 min_registered_years=1,
                 min_events=None,
                 ):

        match index_on:
            case str() | list():
                self._index_on_fn = self._set_event_index_date
                self._index_on_event = index_on
            case float() | int():
                assert age_at_entry_range[0] <= index_on <= age_at_entry_range[
                    1], f"Age {index_on} is not within the range {age_at_entry_range}"
                self._index_on_fn = self._set_age_index_date
                self._index_on_age = index_on
            case _:
                raise ValueError(f"Unsupported type for index_on: {type(index_on)}")

        self._outcomes = outcomes
        self._require_outcome = require_outcome
        self._include_on_events_prior_to_index = include_on_events_prior_to_index
        self._exclude_on_events_prior_to_index = exclude_on_events_prior_to_index
        self._exclude_on_events = exclude_on_events
        self._study_period = study_period
        self._age_at_entry_range = age_at_entry_range
        self._min_registered_years = min_registered_years
        self._min_events = min_events

    def fit(self,
            lazy_static,
            lazy_combined_frame):

        ###################
        # Reduce the frames by removing any patients who do not satisfy global criteria
        ###################
        lazy_static, lazy_combined_frame = self._remove_on_global_criteria(lazy_static, lazy_combined_frame)
        lazy_static = lazy_static.collect().lazy()  # Force collection (TODO: remove)
        lazy_combined_frame = lazy_combined_frame.collect().lazy()

        ###################
        # Set an index date
        ###################
        lazy_static, lazy_combined_frame = self._index_on_fn(lazy_static, lazy_combined_frame)
        lazy_static = lazy_static.collect().lazy()  # Force collection (TODO: remove)
        lazy_combined_frame = lazy_combined_frame.collect().lazy()

        ###################
        # Given the index date, reduce to patient satisfying pre- and post- index criteria
        ###################
        lazy_static, lazy_combined_frame = self._reduce_on_index_date(lazy_static, lazy_combined_frame)

        ###################
        # Given this index date, reduce events to those leading to and including the date, and the final observation (observed or last seen within study period)
        ###################
        lazy_static, lazy_combined_frame = self._reduce_on_outcome(lazy_static, lazy_combined_frame)

        ###################
        # Optionally remove patients who did not experience more than self._min_events
        ###################
        lazy_static, lazy_combined_frame = self._reduce_on_dynamic_events(lazy_static, lazy_combined_frame)

        # patient_id_for_checking = 5922416221434
        # print(lazy_static.filter(pl.col("PATIENT_ID")==patient_id_for_checking).collect())
        # print(lazy_combined_frame.filter(pl.col("PATIENT_ID")==patient_id_for_checking).sort(["PRACTICE_ID", "PATIENT_ID", "DATE"]).collect())

        return lazy_static, lazy_combined_frame

    def _remove_on_global_criteria(self,
                                   lazy_static,
                                   lazy_combined_frame):
        """
            Lazy frame reduction step: Remove patients who do not fit the global critera defined by:
             * exclude_on_events
             * minimum registered period
        """

        # Exclude patients who have an `exclude_on_events` event
        if self._exclude_on_events is not None:
            # Get the patients who have any of these events, regardless of when
            patients_without_excluded_events = (
                lazy_combined_frame
                .with_columns(pl.col("EVENT").is_in(self._exclude_on_events).alias("IS_EXC_EVENT"))
                .groupby("PATIENT_ID").agg(pl.col("IS_EXC_EVENT").sum())
                .filter(pl.col("IS_EXC_EVENT") == 0)
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )
            # and reduce original frames using this list
            lazy_combined_frame = patients_without_excluded_events.join(lazy_combined_frame, on=["PATIENT_ID"],
                                                                        how="inner")
            lazy_static = patients_without_excluded_events.join(lazy_static, on=["PATIENT_ID"], how="inner")

            # Remove all patients who are not registered at the practice for a period of at least `min_registered_years`
        lazy_static = (
            lazy_static
            .select([
                (pl.col("END_DATE") - pl.col("START_DATE")).dt.days().alias("DAYS_REGISTERED"), "*"
            ])
            .filter(pl.col('DAYS_REGISTERED') >= self._min_registered_years * 365.25)
            .drop("DAYS_REGISTERED")
        )
        patients_within_registration_length = lazy_static.select(pl.col('PATIENT_ID'))
        lazy_combined_frame = patients_within_registration_length.join(lazy_combined_frame, on=["PATIENT_ID"],
                                                                       how="inner")

        return lazy_static, lazy_combined_frame

    def _set_age_index_date(self,
                            lazy_static,
                            lazy_combined_frame):
        """
        Set the index date to time at which patient reaches a certain age.

        Ensuring that the index date is within the study period - otherwise remove from cohort.
        Ensure that the index age is between minimum and maximum age at cohort entry.
        """

        # Get the date at which they turn specified age, and set the date as the index date
        # This is the earliest possible index date. It would be later if this lies outside of study dates
        patients_with_index_age = (
            lazy_static
            .with_columns((pl.col("YEAR_OF_BIRTH") + pl.duration(days=self._index_on_age * 365.25)).alias("INDEX_DATE"))
        )

        # Reduce this list to include only patients with the index age occuring during the study period
        # after study start date
        # and before study end date
        patients_with_index_age = (
            patients_with_index_age
            .filter(
                pl.col('INDEX_DATE') >= pl.lit(self._study_period[0])
                .str.strptime(pl.Date, fmt="%F")
            )
            .filter(
                pl.col('INDEX_DATE') <= pl.lit(self._study_period[1])
                .str.strptime(pl.Date, fmt="%F")
            )
        )

        # # and include only those with the age is within the specified age range - this is a sanity check
        # patients_with_index_age = (
        #     patients_with_index_age
        #     .filter(pl.col('DAYS_SINCE_BIRTH') >= self._age_at_entry_range[0]*365.25)               # index event occurred after minimum age
        #     .filter(pl.col('DAYS_SINCE_BIRTH') <= self._age_at_entry_range[1]*365.25)               # index event occurred before maximum age
        # )

        # Get this patient list
        patients_with_index_age = (
            patients_with_index_age
            .select(pl.col('PATIENT_ID', "INDEX_DATE"))
        )

        # and reduce original frames using this list, adding the new index date to both frames
        lazy_combined_frame = (
            patients_with_index_age
            .join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")
        )
        lazy_static = (
            patients_with_index_age
            .join(lazy_static.drop("INDEX_DATE"), on=["PATIENT_ID"], how="inner")
        )

        return lazy_static, lazy_combined_frame

    def _set_event_index_date(self,
                              lazy_static,
                              lazy_combined_frame):
        """
        Set the index date to the first of a potential list of outcome events.

        Ensure the index date occurs within the study period, or otherwise remove from cohort.
        Ensure that the index date is between minimum and maximum age at cohort entry.
        """

        # Retain only patients with the required events occurring (at any time)
        # Include only patients who experienced the events
        if type(self._index_on_event) is list:
            patients_with_index_event = (
                lazy_combined_frame
                .filter(pl.col("EVENT").is_in(self._index_on_event))
            )
        else:
            patients_with_index_event = (
                lazy_combined_frame
                .filter(pl.col("EVENT") == self._index_on_event)
            )

        # Reduce this list to include only patients with the required events occuring during the study period
        # and before study end date
        # after study start date
        patients_with_index_event = (
            patients_with_index_event
            .filter(
                pl.col('DATE') >= pl.lit(self._study_period[0])
                .str.strptime(pl.Date, fmt="%F")
            )
            .filter(
                pl.col('DATE') <= pl.lit(self._study_period[1])
                .str.strptime(pl.Date, fmt="%F")
            )
        )

        # and include only those with these events within the specified age range
        # index event occurred after minimum age
        # index event occurred before maximum age
        patients_with_index_event = (
            patients_with_index_event
            .filter(
                pl.col('DAYS_SINCE_BIRTH') >= self._age_at_entry_range[0] * 365.25
            )
            .filter(
                pl.col('DAYS_SINCE_BIRTH') <= self._age_at_entry_range[1] * 365.25
            )
        )

        # Get the first valid index event if multiple exist (e.g. repeat diagnosis, multiple index events considered), and set the date as the index date
        # This is the earliest possible index date. It would be later if this lies outside of study dates
        patients_with_index_event = (
            patients_with_index_event
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])  # Sort to ensure date order within patients
            .unique(subset=["PRACTICE_ID", "PATIENT_ID"],
                    keep="first")  # Keep chronologically first required event experienced by patient
            .with_columns((pl.col('DATE').alias('INDEX_DATE')))
        )

        # Get this patient list
        patients_with_index_event = (
            patients_with_index_event
            .unique("PATIENT_ID")
            .select(pl.col('PATIENT_ID', "INDEX_DATE"))
        )

        # and reduce original frames using this list, adding the new index date to both frames
        lazy_combined_frame = patients_with_index_event.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")
        lazy_static = patients_with_index_event.join(lazy_static.drop("INDEX_DATE"), on=["PATIENT_ID"], how="inner")

        return lazy_static, lazy_combined_frame

    def _reduce_on_index_date(self,
                              lazy_static,
                              lazy_combined_frame, ):

        #############################################################
        # Remove patients where `exclude_on_events_prior_to_index`
        # occured before the index event
        #############################################################

        # Get the patients without any of the `_exclude_on_events_prior_to_index` events occurring before the index date
        # For example, if we are interested in the possibility of prescribing a medicine then we may want to exclude those already on the medicine
        # TODO: this can be optimised further
        if self._exclude_on_events_prior_to_index is not None:
            patients_without_excluded_prior_events = (
                lazy_combined_frame
                .filter(pl.col('DATE') < pl.col('INDEX_DATE'))
                .with_columns(pl.col("EVENT").is_in(self._exclude_on_events_prior_to_index).alias("IS_EXC_EVENT"))
                .groupby("PATIENT_ID")
                .agg(pl.col("IS_EXC_EVENT").sum())
                .filter(pl.col("IS_EXC_EVENT") == 0)
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )

            # and reduce original frames using this list
            lazy_combined_frame = patients_without_excluded_prior_events.join(lazy_combined_frame, on=["PATIENT_ID"],
                                                                              how="inner")
            lazy_static = patients_without_excluded_prior_events.join(lazy_static, on=["PATIENT_ID"], how="inner")

            # Get the patients with any of the `_include_on_events_prior_to_index` events occurring before the index date
        # For example, if we are interested in the effect of prescribing a medicine for a diagnosis, then we may want to include those with the diagnosis
        # TODO: this can be optimised further
        if self._include_on_events_prior_to_index is not None:
            patients_with_included_prior_events = (
                lazy_combined_frame
                .filter(pl.col('DATE') < pl.col('INDEX_DATE'))  # keep only events before index date
                # .select([
                # (pl.col("INDEX_DATE") - pl.col("DATE")).dt.days().alias("DAYS_BEFORE_INDEX"), "*"
                # ])
                .filter((pl.col("INDEX_DATE") - pl.col("DATE")).dt.days() <= self._include_on_events_prior_to_index[
                    1])  # and less than X days before index date
                .with_columns((pl.col("EVENT") == self._include_on_events_prior_to_index[0]).alias(
                    "IS_INC_EVENT"))  # create a new column indicating inclusion events
                .groupby("PATIENT_ID")
                .agg(pl.col("IS_INC_EVENT").sum())  # and count how many inclusion events per patient
                .filter(pl.col("IS_INC_EVENT") > 0)  # take all patients with inclusion events
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )

            # and reduce original frames using this list
            lazy_combined_frame = patients_with_included_prior_events.join(lazy_combined_frame, on=["PATIENT_ID"],
                                                                           how="inner")
            lazy_static = patients_with_included_prior_events.join(lazy_static, on=["PATIENT_ID"], how="inner")

        return lazy_static, lazy_combined_frame

    def _reduce_on_outcome(self,
                           lazy_static,
                           lazy_combined_frame, ):

        #############################################################
        # Retain only patients with valid events following the index date
        # ... as we want followup predictions, we need to ensure an
        #     event occurs - even if this is not an outcome
        #############################################################

        # Keep only patients with any event occurring between index and study end
        #   (i.e. remove patients where index event is last event in study)
        patients_with_target = (
            lazy_combined_frame
            .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
            .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))
            .select(pl.col('PATIENT_ID'))  # get the patient list
            .unique()
        )
        patients_with_target_list = patients_with_target.collect().to_series().to_list()

        lazy_combined_frame = lazy_combined_frame.filter(pl.col("PATIENT_ID").is_in(patients_with_target_list))
        lazy_static = lazy_static.filter(pl.col("PATIENT_ID").is_in(patients_with_target_list))

        #############################################################
        # GET OUTCOMES
        # or (by default) if no outcome, the last seen event within study period
        #############################################################

        # Get all possible outcomes
        # If multiple outcomes follow index, we take all for now, but later reduce this to the first seen.
        if not callable(self._outcomes):
            lazy_combined_frame_outcomes = lazy_combined_frame.filter(pl.col("EVENT").is_in(self._outcomes))
        else:
            lazy_combined_frame_outcomes = lazy_combined_frame.filter(
                pl.col("EVENT").apply(self._outcomes, return_dtype=pl.Boolean)
            )

        # Retain those occurring after index, and before study end
        # Only look at outcomes within study period
        lazy_combined_frame_outcomes = (
            lazy_combined_frame_outcomes
            .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
            .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))
        )

        # Get outcomes
        if self._require_outcome is False:
            # If we do not require the outcome to have occurred - for example in survival cases where we are also interested in right-censored events
            # For example:
            #    if we are predicting the outcome of Hypertension, we are still interested in cases where an event has not occurred `yet`, but may still occur
            #    e.g. a patient has experienced the index event, hypertension hasn't occurred within study period, but there is knowledge in knowing that
            #    it has not occurred by the end of the study period.

            # Get last observation within study period
            # Keep chronologically last event experienced by patient
            lazy_combined_frame_last_event_in_study = (
                lazy_combined_frame
                .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date,
                                                                                     fmt="%F"))  # Only look at events within study period
                .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])  # Sort to ensure date order within patients
                .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="last")
            )

            # Take first between these - this will be the first outcome if one has occurred, otherwise the last event
            # Keep chronologically last event experienced by patient
            outcome = (
                pl.concat([lazy_combined_frame_outcomes, lazy_combined_frame_last_event_in_study])
                .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])  # Sort to ensure date order within patients
                .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")
            )

        else:

            # Take first of the valid outcomes
            # Keep chronologically last event experienced by patient
            outcome = (
                lazy_combined_frame_outcomes
                .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])  # Sort to ensure date order within patients
                .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")
            )

            # Unlike the above case, where we know every patient will have a valid outcome, here some will now be excluded.
            # Remove these patients from the `lazy_combined_frame` and `lazy_static` lazy frames
            patients_with_valid_outcome = (
                outcome
                .select(pl.col('PATIENT_ID'))  # get the patient list
            )
            patients_with_valid_outcome_list = patients_with_valid_outcome.collect().to_series().to_list()

            lazy_combined_frame = lazy_combined_frame.filter(
                pl.col("PATIENT_ID").is_in(patients_with_valid_outcome_list))
            lazy_static = lazy_static.filter(pl.col("PATIENT_ID").is_in(patients_with_valid_outcome_list))

            # print(patients_with_valid_outcome_list)
            # print(lazy_static.collect())
            # print(lazy_combined_frame.collect())
            # assert 1 ==0

        #############################################################
        # MERGE EVENTS UP TO AND INCLUDING INDEX, AND OUTCOME
        #############################################################

        # Get events which occured before index
        lazy_combined_frame_before_index = (
            lazy_combined_frame
            .filter(pl.col('DATE') <= pl.col('INDEX_DATE'))
        )

        # and merge this with the events that occurred up to and including the index from the last section
        new_combined_frame = (
            pl.concat([lazy_combined_frame_before_index, outcome])
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])  # Sort to ensure date order within patients
        )

        return lazy_static, new_combined_frame

    def _reduce_on_dynamic_events(self,
                                  lazy_static,
                                  lazy_combined_frame, ):

        #############################################################
        # Remove patients with fewer than self._min_events events up to and including the index event.
        #############################################################

        minimum_dynamic_events = self._min_events if self._min_events is not None else 1

        patients_with_min_events = (
            lazy_combined_frame
            .filter(pl.col('DATE') <= pl.col('INDEX_DATE'))  # Look at first N-1 events (exclude outcome)
            .with_columns(pl.col("EVENT"))
            .groupby("PATIENT_ID")
            .agg(pl.col("EVENT").count())  # Count the number of events experienced
            .filter(pl.col("EVENT") >= minimum_dynamic_events)  # Only include patients with minimum requirement
            .unique("PATIENT_ID")
            .select(pl.col('PATIENT_ID'))
        )

        # and reduce original frames using this list
        lazy_combined_frame = patients_with_min_events.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")
        lazy_static = patients_with_min_events.join(lazy_static, on=["PATIENT_ID"], how="inner")

        return lazy_static, lazy_combined_frame
