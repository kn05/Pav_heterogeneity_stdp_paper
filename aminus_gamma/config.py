import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, "..", "1_data")
result_dir = os.path.join(script_dir, "..", "2_result")
derived_param_dir = os.path.join(script_dir, "1_derived_parameter")

if "SUBFOLDER_NAME" in os.environ:
    subfolder_name = os.environ.get("SUBFOLDER_NAME")
    data_subfolder_dir = os.path.join(data_dir, subfolder_name)
    data_record_dir = os.path.join(data_subfolder_dir, "record")
    data_csv_dir = os.path.join(data_subfolder_dir, "csv")
    result_subfolder_dir = os.path.join(result_dir, subfolder_name)
