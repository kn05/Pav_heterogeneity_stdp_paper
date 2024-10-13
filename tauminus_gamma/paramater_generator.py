import sys
import os
import json
from config import *
from common import (
    get_params,
)

params = get_params(build_model=True, measure_timing=False, use_genn_recording=True)
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    t1 = float(sys.argv[2])
    t2 = float(sys.argv[3])
    t3 = float(sys.argv[4])
    t4 = float(sys.argv[5])
    params["seed"] = seed
    (
        params["tauPlus_a"],
        params["tauPlus_b"],
        params["tauMinus_a"],
        params["tauMinus_b"],
    ) = (t1, t2, t3, t4)
print(f"folder:{subfolder_name}, seed:{params['seed']}")

# Create subfolder and its subdirectories
os.makedirs(data_subfolder_dir, exist_ok=True)
os.makedirs(data_record_dir, exist_ok=True)
os.makedirs(data_csv_dir, exist_ok=True)
os.makedirs(result_subfolder_dir, exist_ok=True)

"""with open(params["cwd"] + "/../" + "params_control_input.json") as json_file:
    params_control_input = json.load(json_file)
    params.update(params_control_input)"""

# write json data to params.json
with open(os.path.join(data_subfolder_dir, "params.json"), "w") as fp:
    json.dump(params, fp, indent=4)
