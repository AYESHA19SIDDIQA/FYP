from .sc_pipeline import get_scratch_pl
from .sc_pipeline import get_scratch_updated_pl

import numpy as np

tuh_dataset = r'/media/tukl/ee279b7d-bb8a-4a20-8bf9-90b2c542efcc/EEG Datasets/TUH EEG Corpus'
nmt_dataset = r'E:\NMT_events\NMT_events\edf'
nmt_new_dataset = r'E:\NMT_events\NMT_events\edf'

def get_def_ds(mins = 1):
    return (tuh_dataset, get_scratch_pl('TUH', mins), np.array([1, 1]), "results/tuh"), \
            (nmt_dataset, get_scratch_updated_pl('NMT', mins), np.array([1900, 305]), "results/nmt"), \
            (nmt_new_dataset, get_scratch_updated_pl('NMT', mins), np.array([3945, 708]), "results/nmt4k")

