##############################################################################

#           surrogates
#           Andy Kwak

##############################################################################

#       Fake markets for honest controls. clock_shuffle keeps the trading
#       day's schedule exactly (every time-of-day slot keeps its own values,
#       so the open/close volume U-shape survives) but shuffles which day
#       each value came from, destroying any rhythm that is not the clock.
#       Anything we "hear" in real data must beat what we hear in these.


#       @requires a frame indexed by exchange-local bar timestamps
#       @ensures same index, chosen columns permuted within each slot

##############################################################################

import numpy as np


def clock_shuffle(feats, cols, rng):
    #@requires cols are columns of feats; rng is a numpy Generator
    #@ensures \result has feats' index; within each time-of-day slot the
    #         rows of `cols` are permuted across dates (moving together);
    #         every other column is unchanged
    out = feats.copy()
    values = feats[list(cols)].to_numpy()
    shuffled = values.copy()
    slots = feats.index.time
    for slot in np.unique(slots):
        pos = np.flatnonzero(slots == slot)
        shuffled[pos] = values[rng.permutation(pos)]
    out[list(cols)] = shuffled
    return out
