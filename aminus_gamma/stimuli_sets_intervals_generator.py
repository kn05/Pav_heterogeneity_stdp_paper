import os
import json
import numpy as np
import random
from config import *

# import default parameters
with open(os.path.join(data_subfolder_dir, "params.json")) as params_file:
    params = json.load(params_file)
target_stimuli_set_file = os.path.join(data_subfolder_dir, "target_stimuli_set.txt")
stimuli_sets_intervals_file = os.path.join(
    data_subfolder_dir, "stimuli_sets_intervals.txt"
)

dt1 = params["dt1"]
dt2 = params["dt2"]

# generate stimuli_sets_intervals
target_stimuli_set = np.array([[0, 1, 2, dt1, dt2]])

# generate target_stimuli_set
# generate [0, 1, 2, *, *]
stimuli_sets_intervals = [
    np.array([0, 1, 2, d1, d2]) for d1 in range(2 * dt1) for d2 in range(2 * dt2)
]
# generate [0, 1, *, dt1, dt2]
for i in range(0, params["num_stimuli_sets"], 1):
    stimuli_sets_intervals.append(np.array([0, 1, i, dt1, dt2]))
# generate [0, 1, x, *, 0]
for d1 in range(0, 2 * dt1, 1):
    stimuli_sets_intervals.append(np.array([0, 1, params["num_stimuli_sets"], d1, 0]))
# generate [0, x, 2, 0, *]
for d2 in range(0, 2 * dt2, 1):
    stimuli_sets_intervals.append(np.array([0, params["num_stimuli_sets"], 2, 0, d2]))
# only stim subgroup 0 or 1 or 2
stimuli_sets_intervals.append(
    np.array(
        [
            0,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            dt1,
            dt2,
        ]
    )
)
stimuli_sets_intervals.append(
    np.array(
        [
            1,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            dt1,
            dt2,
        ]
    )
)
stimuli_sets_intervals.append(
    np.array(
        [
            2,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            dt1,
            dt2,
        ]
    )
)
# generate [4, 5, 6, dt1, dt2]
stimuli_sets_intervals.append(np.array([4, 5, 6, dt1, dt2]))
# generate [4, 5, 6, dt1, dt2]
stimuli_sets_intervals.append(np.array([34, 8, 71, dt1, dt2]))
# some random arrays with dt1, dt2
for i in range(100):
    stimuli_sets_intervals.append(
        np.array(
            [
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                dt1,
                dt2,
            ]
        )
    )
# some random arrays
for i in range(100):
    stimuli_sets_intervals.append(
        np.array(
            [
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, 2 * dt1 + 1),
                random.randint(0, 2 * dt2 + 1),
            ]
        )
    )

np.savetxt(target_stimuli_set_file, target_stimuli_set, fmt="%d")
np.savetxt(stimuli_sets_intervals_file, stimuli_sets_intervals, fmt="%d")
