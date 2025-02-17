import polars as pl
import logging
from FastEHR.dataloader.utils.study_criteria import index_inclusion_method

def t2d_inclusion_method(index_on_event="TYPE2DIABETES",
                         outcomes=["IHDINCLUDINGMI_OPTIMALV2",
                                   "ISCHAEMICSTROKE_V2",
                                   "MINFARCTION",
                                   "STROKEUNSPECIFIED_V2",
                                   "STROKE_HAEMRGIC"],
                         exclude_on_events=["TYPE1DM"],
                         exclude_on_events_prior_to_index=['Statins'],
                         study_period=["1998-01-01", "2019-12-31"],
                         age_at_entry_range=[25, 85],
                         min_registered_years=1,
                         min_events=None,
                         ):
    T2D_inclusion = index_inclusion_method(index_on=index_on_event,
                                           outcomes=outcomes,
                                           exclude_on_events_prior_to_index=exclude_on_events_prior_to_index,
                                           exclude_on_events=exclude_on_events,
                                           study_period=study_period,
                                           age_at_entry_range=age_at_entry_range,
                                           min_registered_years=min_registered_years,
                                           min_events=min_events,
                                           )
    return T2D_inclusion.fit


def hypertension_inclusion_method(index_on_event=["ACE_Inhibitors_D2T",
                                                  "ARBs_Luyuan",
                                                  "BetaBlockers_OPTIMAL",
                                                  "CalciumChannelBlck_D2T",
                                                  "Thiazide_Diuretics_v2",
                                                  # "All_Diuretics_D2T",                       # Not using these as not all diuretics are used for BP lowering
                                                  # "All_Diuretics_ExclLactones_D2T",
                                                  "AlphaBlocker",
                                                  "AldosteroneAntagonist_D2T"
                                                  # Used in extremely treatment resistant HTN
                                                  ],
                                  outcomes=["Systolic_blood_pressure_4"],
                                  require_outcome=True,
                                  # Remove right-censoring of patients who do not observe the outcome
                                  include_on_events_prior_to_index=("HYPERTENSION", 30 * 2),
                                  # Must have had a hypertension diagnosis in the 60 days prior to index date
                                  exclude_on_events=None,
                                  exclude_on_events_prior_to_index=None,
                                  study_period=["1998-01-01", "2019-12-31"],
                                  age_at_entry_range=[25, 85],
                                  min_registered_years=1,
                                  min_events=None,
                                  ):
    hyp_inclusion = index_inclusion_method(index_on=index_on_event,
                                           outcomes=outcomes,
                                           require_outcome=require_outcome,
                                           include_on_events_prior_to_index=include_on_events_prior_to_index,
                                           exclude_on_events_prior_to_index=exclude_on_events_prior_to_index,
                                           exclude_on_events=exclude_on_events,
                                           study_period=study_period,
                                           age_at_entry_range=age_at_entry_range,
                                           min_registered_years=min_registered_years,
                                           min_events=min_events,
                                           )
    return hyp_inclusion.fit


def multimorbidity_inclusion_method(index_on_age=50,
                                    study_period=["1998-01-01", "2019-12-31"],
                                    age_at_entry_range=[25, 85],
                                    min_registered_years=1,
                                    min_events=2,
                                   ):

    # Custom function which is used as a filter on the dynamic frame to select outcomes as events which are a diagnosis.
    #  In our CPRD example, all diagnoses are coded as full capitals, so we can distinguish them this way
    def custom_mm_filter(s: str) -> bool:
        return s.isupper()          # Keep only strings with all capital letters

    mm_inclusion = index_inclusion_method(index_on=index_on_age,
                                           outcomes=custom_mm_filter,
                                           study_period=study_period,
                                           age_at_entry_range=age_at_entry_range,
                                           min_registered_years=min_registered_years,
                                           min_events=min_events,
                                           )
    return mm_inclusion.fit
