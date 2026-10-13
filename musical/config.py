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
