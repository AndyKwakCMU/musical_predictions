##############################################################################

#           config
#           Andy Kwak

##############################################################################

#       Shared experiment settings. DEV_END marks the held-out test period:
#       nothing on or after it is used until the final Test C run.

##############################################################################

DEV_END = "2024-01-01"      # bars >= this are the untouched final test set
RAW_DIR = "data/raw"

BARRIER_K = 2.0             # barriers at entry +/- k * ATR
HORIZON = 10                # bars before the time stop

WINDOW = 64                 # bars of history fed to the CNN
DATASET_DIR = "data/datasets"

# Test B splits (all inside the dev period)
VAL_START = "2018-01-01"
TEST_START = "2021-01-01"
MODEL_DIR = "models"

# Intraday (IBKR 15m bars). Settings fixed 2026-10-13 BEFORE any intraday
# results were seen; see docs/experiments.md.
INTRADAY_RAW_DIR = "data/raw_ib"
INTRADAY_SEASONAL_DAYS = 40


def tsa_config(interval):
    #@ensures the TSA settings for this bar interval (daily vs intraday)
    from musical.tsa import TSAConfig
    if interval == "1d":
        return TSAConfig()
    return TSAConfig(seasonal_days=INTRADAY_SEASONAL_DAYS)


def raw_dir(interval):
    #@ensures where protocol CSVs for this interval live
    return RAW_DIR if interval == "1d" else INTRADAY_RAW_DIR

# Intraday Test B splits (data starts 2016; holdout is still >= DEV_END)
INTRADAY_VAL_START = "2021-01-01"
INTRADAY_TEST_START = "2023-01-01"
