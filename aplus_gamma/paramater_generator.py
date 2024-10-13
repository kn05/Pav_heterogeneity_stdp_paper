import sys
import os
import json
from config import *
from common import (
    get_params,
)

params = get_params(build_model=True, measure_timing=False, use_genn_recording=True)
if len(sys.argv) > 1:
    params["seed"] = int(sys.argv[1])
    params["dt1"] = int(sys.argv[2])
    params["dt2"] = int(sys.argv[3])
    params["aPlus_a"] = float(sys.argv[4])
    params["aPlus_b"] = float(sys.argv[5])
    params["aMinus_a"] = float(sys.argv[6])
    params["aMinus_b"] = float(sys.argv[7])


print(f"folder:{subfolder_name}")
print(f"sys.argv:{sys.argv}")

# Create subfolder and its subdirectories
os.makedirs(data_subfolder_dir, exist_ok=True)
os.makedirs(data_record_dir, exist_ok=True)
os.makedirs(data_csv_dir, exist_ok=True)
os.makedirs(result_subfolder_dir, exist_ok=True)

# write json data to params.json
with open(os.path.join(data_subfolder_dir, "params.json"), "w") as fp:
    json.dump(params, fp, indent=4)
