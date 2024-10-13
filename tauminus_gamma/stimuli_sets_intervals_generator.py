import os
import json
import numpy as np
import random
from config import *

# set file paths
params_file = os.path.join(script_dir, "0_primitive_parameter/params.json")
target_stimuli_set_file = os.path.join(
    script_dir, "1_derived_parameter/target_stimuli_set.txt"
)
stimuli_sets_intervals_file = os.path.join(
    script_dir, "1_derived_parameter/stimuli_sets_intervals.txt"
)

# import default parameters
with open(params_file, "r") as f:
    params = json.load(f)

# generate stimuli_sets_intervals
target_stimuli_set = np.array([[0, 1, 2, 3, 3]])

# generate target_stimuli_set
# generate [0, 1, 2, *, *]
stimuli_sets_intervals = [
    np.array([0, 1, 2, d1, d2]) for d1 in range(6) for d2 in range(6)
]
# generate [0, 1, 2, 3, 3]
for i in range(0, params["num_stimuli_sets"], 1):
    stimuli_sets_intervals.append(np.array([0, 1, i, 3, 3]))
# generate [0, 1, x, *, 0]
for d1 in range(0, 6, 1):
    stimuli_sets_intervals.append(np.array([0, 1, params["num_stimuli_sets"], d1, 0]))
# generate [0, x, 2, 0, *]
for d2 in range(0, 6, 1):
    stimuli_sets_intervals.append(np.array([0, params["num_stimuli_sets"], 2, 0, d2]))
# only stim subgroup 0 or 1 or 2
stimuli_sets_intervals.append(
    np.array(
        [
            0,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
stimuli_sets_intervals.append(
    np.array(
        [
            1,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
stimuli_sets_intervals.append(
    np.array(
        [
            2,
            params["num_stimuli_sets"],
            params["num_stimuli_sets"],
            params["dt1"],
            params["dt2"],
        ]
    )
)
# generate [4, 5, 6, dt1, dt2]
stimuli_sets_intervals.append(np.array([4, 5, 6, params["dt1"], params["dt2"]]))
# some random arrays
for i in range(100):
    stimuli_sets_intervals.append(
        np.array(
            [
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, params["num_stimuli_sets"]),
                random.randint(0, 5),
                random.randint(0, 5),
            ]
        )
    )

np.savetxt(target_stimuli_set_file, target_stimuli_set, fmt="%d")
np.savetxt(stimuli_sets_intervals_file, stimuli_sets_intervals, fmt="%d")
