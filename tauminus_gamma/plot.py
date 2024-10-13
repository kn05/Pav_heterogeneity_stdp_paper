# %%
# code block for test run
#
"""import os
import inquirer

env = os.environ
env["SUBFOLDER_NAME"] = "data00"
env["SIGMA"] = "0.5"
from config import *


def get_plot_files_from_file(file_path):
    with open(file_path, "r") as f:
        plot_files = [
            line.strip() for line in f if line.strip()
        ]  # 파일에서 각 줄을 읽어와 공백 제거 후 리스트로 변환
    return plot_files


# 플롯할 파일 목록을 선택하도록 질문 설정
plot_file_path = os.path.join(
    script_dir, "plot_file.txt"
)  # 플롯 파일 목록이 담긴 텍스트 파일 경로
file_choices = get_plot_files_from_file(plot_file_path)

plot_files = ",".join(file_choices)
#
env["PLOT_FILES"] = plot_files
"""

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from six import iteritems
import scipy as sp
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
from collections import Counter


from common import (
    get_params,
    convert_spikes,
)

# %%
from config import *


# %%
def read_spikes(filename):
    return np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
        dtype={"names": ("time", "id"), "formats": (float, np.int64)},
    )


# %%
def get_masks(times, params):
    return (
        np.where(times < params["record_time_ms"]),
        np.where(times > (params["duration_ms"] - params["record_time_ms"])),
    )


# %%
def MSE(y, yhat):
    return np.mean((y - yhat) ** 2)


# %%
def RMSE(y, yhat):
    return np.sqrt(MSE(y, yhat))


# %%
def NRMSE(y, yhat):
    ybar = np.mean(y)
    return RMSE(y, yhat) / ybar


# %%
def weight_in_out_plot(input_numpy_type_peak_t1_t2ay):
    indegree = input_numpy_type_peak_t1_t2ay.sum(axis=0)
    outdegree = input_numpy_type_peak_t1_t2ay.sum(axis=1)
    return indegree, outdegree


# %%
def hist_spike(sT, t0, tmax):
    counter = Counter(sT)
    full_range = np.arange(t0, tmax + 1)
    # 각 숫자의 빈도를 포함하도록 데이터 준비
    full_counter = {key: counter.get(key, 0) for key in full_range}
    # 데이터를 x, y로 분리
    x = np.array(list(full_counter.keys()))
    y = np.array(list(full_counter.values()))
    return x, y


# %%
def get_sdf_masks(times, sigma, params):
    return (
        np.where(
            (times > (3.0 * sigma))
            & (times < (params["record_time_ms"] - (3.0 * sigma)))
        ),
        np.where(
            (times > (params["duration_ms"] - params["record_time_ms"] + (3.0 * sigma)))
            & (times < (params["duration_ms"] - (3.0 * sigma)))
        ),
    )


# %%
def make_sdf(sT, t0, tmax, dt, sigma):
    time = np.round(np.arange(t0 - 3 * sigma, tmax + 3 * sigma, dt), 3)
    sdfs = np.zeros_like(time)

    kwdt = np.round(3 * sigma, 3)
    x = np.arange(-kwdt, kwdt, dt)
    x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
    x = x / (sigma * np.sqrt(2.0 * np.pi))

    if sT is not None:
        for t in sT:
            left = int((t - t0 - kwdt) / dt)
            right = int((t - t0 + kwdt) / dt)
            sdfs[left:right] += x
    return time, sdfs


# %%
def arr_shift(arr, num_shifts):
    length = len(arr)
    shifted_arr = np.zeros(length, dtype=arr.dtype)
    if num_shifts > 0:
        num_shifts = num_shifts % length
        shifted_arr[num_shifts:] = arr[: length - num_shifts]
    elif num_shifts < 0:
        num_shifts = (-num_shifts) % length
        shifted_arr[: length - num_shifts] = arr[num_shifts:]
    else:
        shifted_arr = arr
    return shifted_arr


# %%
import json

f_params = open(f"{data_subfolder_dir}/params.json", encoding="UTF-8")
params = json.loads(f_params.read())
sigma = float(os.environ.get("SIGMA"))  # params["sigma"] or 1.0
print(f"sigma: {sigma}")

plot_files = os.environ.get("PLOT_FILES")

os.makedirs(result_subfolder_dir, exist_ok=True)

indicator_path = os.path.join(result_subfolder_dir, "sdf_indicator.json")
if os.path.exists(indicator_path):
    with open(indicator_path, "r") as f:
        data = json.load(f)
else:
    data = {}
data_update_tag = False

# %%
# "SDF" is a variable that determines whether to plot things related to sdf.
sdf_files = [
    "SDF_TARGET",
    "SDF_INDICATORS",
    "SDF_VERSUS",
    "SDF_PROFILES",
    "SDF_PROFILES_0XX_1XX_2XX",
    "SDF_CORR_HEATMAP",
    "SDF_NORMAL_CORR_HEATMAP" "PEAK_SHIFT_HEATMAP",
]

# always
if "seed" in params:
    np.random.seed(params["seed"])
num_neurons = 1000
# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
input_sets = [
    np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
    for _ in range(params["num_stimuli_sets"])
]

if any(sdf in plot_files for sdf in sdf_files):
    e_spikes = read_spikes(f"{data_csv_dir}/izhikevich_e_spikes.csv")
    i_spikes = read_spikes(f"{data_csv_dir}/izhikevich_i_spikes.csv")

    # Read stimuli sets
    target_stimuli_set = np.loadtxt(
        os.path.join(derived_param_dir, "target_stimuli_set.txt"), dtype=int
    )
    stimuli_sets_intervals = np.loadtxt(
        os.path.join(derived_param_dir, "stimuli_sets_intervals.txt"), dtype=int
    )
    # Read stimuli
    stimuli = np.loadtxt(
        f"{data_csv_dir}/izhikevich_stimulus_times.csv",
        delimiter=",",
        dtype={
            "names": ("time", "id1", "id2", "id3", "t1", "t2"),
            "formats": (float, np.int64, np.int64, np.int64, np.int64, np.int64),
        },
    )
    # Read rewards
    reward_times = np.loadtxt(
        f"{data_csv_dir}/izhikevich_reward_times.csv", dtype=float
    )

    # make second sdf
    spikes = np.concatenate((e_spikes["time"], i_spikes["time"]))
    total_second_hist = hist_spike(spikes, 0, params["duration_ms"])
    # sdf time resolution
    sdf_dt = 0.1
    sdf_spikes = np.concatenate(
        (
            e_spikes["time"],
            i_spikes["time"],
        )
    )
    # sdf data in (time, sdf) format
    second_sdf = make_sdf(
        sdf_spikes,
        0.0,
        params["duration_ms"],
        sdf_dt,
        sigma,
    )
    second_sdf = (np.round(second_sdf[0], 2), second_sdf[1])

    # spike sdf dataframe
    stimuli_data = pd.DataFrame(stimuli)
    # (id1, id2, id3, t1, t2)를 튜플로 묶어서 새로운 열(stim_type)로 추가
    stimuli_data["stim_type"] = list(
        zip(
            stimuli_data["id1"],
            stimuli_data["id2"],
            stimuli_data["id3"],
            stimuli_data["t1"],
            stimuli_data["t2"],
        )
    )
    # id1, id2, id3, t1, t2 열 삭제
    stimuli_data.drop(columns=["id1", "id2", "id3", "t1", "t2"], inplace=True)
    # group_tuple과 time을 인덱스로 설정하고 hist 열 추가하여 0으로 채우기
    stimuli_data.set_index(["stim_type", "time"], inplace=True)
    stimuli_data["sdf"] = np.zeros(len(stimuli_data), dtype=object)
    stimuli_data["sdf"] = stimuli_data["sdf"].apply(lambda x: np.zeros(300))

    # fill stimuli_data's sdf column
    for (stim, time), row in stimuli_data.iterrows():
        t = int(time)
        index = np.searchsorted(second_sdf[0], t)
        stimuli_data.loc[(stim, time), "sdf"] = second_sdf[1][index - 20 : index + 280]
    # divide with first and last
    first_stimuli_data = stimuli_data[
        stimuli_data.index.get_level_values("time") < params["record_time_ms"]
    ]
    last_stimuli_data = stimuli_data[
        stimuli_data.index.get_level_values("time")
        > (params["duration_ms"] - params["record_time_ms"])
    ]
    first_stimuli_data = first_stimuli_data[first_stimuli_data["sdf"].apply(len) == 300]
    last_stimuli_data = last_stimuli_data[last_stimuli_data["sdf"].apply(len) == 300]

    # calculate mean of last stimuli's SDF
    mean_last_stimuli_sdf = (
        last_stimuli_data.groupby("stim_type")["sdf"]
        .apply(lambda x: np.mean(x))
        .reset_index(name="mean_sdf")
    )
    # save mean_last_stimuli_sdf to csv
    expanded_mean_sdf = pd.DataFrame(
        mean_last_stimuli_sdf["mean_sdf"].tolist(),
        index=mean_last_stimuli_sdf["stim_type"],
    ).T
    expanded_mean_sdf.to_csv(
        f"{result_subfolder_dir}/mean_last_stimuli_sdf.csv", index=False
    )

# %%
# plot target_stim's SDF
# "SDF_TARGET" : plot (0,1,2,3,3)'s SDF
if "SDF_TARGET" in plot_files:
    fig, axes = plt.subplots()
    target_stim = (0, 1, 2, 3, 3)
    fig.suptitle(f"{target_stim}'s hist")
    selected_row = last_stimuli_data.loc[target_stim]
    series = np.array(selected_row["sdf"].tolist())
    for value in selected_row.values:
        plt.step(np.arange(0, 300), value[0], where="mid", color="blue", alpha=0.4)
    avg = np.mean(series)
    std_dev = np.std(series, axis=0)
    cv = std_dev / avg

    group_data = mean_last_stimuli_sdf.query("stim_type == @target_stim")
    plt.step(
        np.arange(0, 300), group_data["mean_sdf"].values[0], where="mid", color="red"
    )
    fig.savefig(f"{result_subfolder_dir}/sdf_profile_{target_stim}.jpg")

    data.update(
        {
            f"avg_target": avg,
            f"cv_target": np.mean(cv),
        }
    )
    print(f"cv_target: {np.mean(cv)}")
    data_update_tag = True


# %%
# calculate indicators for target_stim vs stand_stim
if "SDF_VERSUS" in plot_files:
    target_stim = (0, 1, 2, 3, 3)
    stand_stim = (4, 5, 6, 3, 3)
    target_data = mean_last_stimuli_sdf.query("stim_type == @target_stim")[
        "mean_sdf"
    ].values[0]
    stand_data = mean_last_stimuli_sdf.query("stim_type == @stand_stim")[
        "mean_sdf"
    ].values[0]
    desired_area = sum(target_data)
    stand_area = sum(stand_data)
    area_ratio = desired_area / stand_area
    area_diff = desired_area - stand_area
    rmse = RMSE(target_data, stand_data)
    nrmse = NRMSE(target_data, stand_data)
    print("area_ratio:", area_ratio)
    print("area_diff:", area_diff)
    print("rmse:", rmse)
    print("nrmse:", nrmse)
    fig, axes = plt.subplots()
    fig.suptitle(f"{target_stim} vs {stand_stim} SDF")
    plt.step(np.arange(0, 300), target_data, where="mid", color="red")
    plt.step(np.arange(0, 300), stand_data, where="mid", color="green")
    plt.legend([f"{target_stim}", f"{stand_stim}"])
    fig.savefig(f"{result_subfolder_dir}/sdf_profile_versus.jpg")
    data.update(
        {
            f"area_ratio": area_ratio,
            f"area_diff": area_diff,
            f"rmse": rmse,
            f"nrmse": nrmse,
        }
    )
    data_update_tag = True

# %%
# plot target sdf profiles
# "SDF_PROFILES" : plot (0,1,2,3,3), (0,1,2,0,0), (0,1,2,5,5), (4,5,6,3,3) SDF
if "SDF_PROFILES" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")
    target_stim = [(0, 1, 2, 3, 3), (0, 1, 2, 0, 0), (0, 1, 2, 5, 5), (4, 5, 6, 3, 3)]
    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]
        peak_index = np.argmax(mean_sdf_values)
        peak_value = mean_sdf_values[peak_index]
        peak_ms = peak_index * sdf_dt
        # Plot profile
        plt.step(
            np.arange(0, 30, sdf_dt), mean_sdf_values, where="mid", label=f"{stim}"
        )
        # Mark the peak with a point and text
        plt.plot(peak_ms, peak_value, "ro")
        plt.text(peak_ms, peak_value, f"({peak_ms:.1f}, {peak_value:.2f})")
    # Set the legend
    plt.legend()
    # Save the graph
    fig.savefig(f"{result_subfolder_dir}/sdf_profiles_target.jpg")

# %%
# plot 0xx 1xx 2xx profile
# "SDF_PROFILES_0XX_1XX_2XX" : plot (0,1,2,0,0), (0,1,2,1,1), (0,1,2,2,2) SDF
if "SDF_PROFILES_0XX_1XX_2XX" in plot_files:
    target_stim = [
        (0, 1, 2, 3, 3),
        (0, 100, 100, 3, 3),
        (1, 100, 100, 3, 3),
        (2, 100, 100, 3, 3),
    ]
    stim_shift = {
        (1, 100, 100, 3, 3): 3,
        (2, 100, 100, 3, 3): 6,
    }

    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]

        # x축을 조정하여 프로파일을 플롯
        shift = stim_shift.get(stim, 0)  # 설정된 밀림 시간 또는 0
        plt.step(
            np.arange(shift, 30, sdf_dt),
            mean_sdf_values[: 300 - int(shift / sdf_dt)],
            where="mid",
            label=f"{stim}",
        )

    # 범례 설정
    plt.legend()
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_profiles_s012.jpg")

    #


if "SDF_PROFILES_0XX_1XX_2XX" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    target_stim = [
        (0, 100, 100, 3, 3),
        (1, 100, 100, 3, 3),
        (2, 100, 100, 3, 3),
    ]

    # 밀어야 할 시간을 설정
    stim_shift = {
        (0, 100, 100, 3, 3): 0,
        (1, 100, 100, 3, 3): 30,
        (2, 100, 100, 3, 3): 60,
    }

    # (0, 100, 100, 3, 3)의 프로파일을 가져오기
    reference_stim = (0, 100, 100, 3, 3)
    reference_data = mean_last_stimuli_sdf.query("stim_type == @reference_stim")
    reference_sdf_values = reference_data["mean_sdf"].values[0]

    # 밀린 프로파일들을 더해 새로운 프로파일을 생성
    new_sdf_values = np.zeros_like(reference_sdf_values)
    sdf_length = len(reference_sdf_values)

    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]

        shift = stim_shift.get(stim, 0)  # 설정된 밀림 시간 또는 0
        shifted_sdf_values = np.zeros_like(mean_sdf_values)

        # 밀림 시간을 반영하여 프로파일을 이동
        if shift > 0:
            shifted_sdf_values[shift:sdf_length] = mean_sdf_values[: sdf_length - shift]
        else:
            shifted_sdf_values = mean_sdf_values

        new_sdf_values += shifted_sdf_values

    # (0, 1, 2, 3, 3)의 프로파일을 가져와 플롯
    original_stim = (0, 1, 2, 3, 3)
    original_data = mean_last_stimuli_sdf.query("stim_type == @original_stim")
    original_sdf_values = original_data["mean_sdf"].values[0]
    plt.step(
        np.arange(0, 30, sdf_dt),
        original_sdf_values,
        where="mid",
        label=f"{original_stim}",
    )

    # 새로운 프로파일 플롯
    plt.step(
        np.arange(0, 30, sdf_dt),
        new_sdf_values,
        where="mid",
        label="New Combined Profile",
    )

    # 범례 설정
    plt.legend()
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_profiles_sum012.jpg")

if "SDF_PROFILES_0XX_1XX_2XX" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    target_stim = [
        (0, 100, 100, 3, 3),
        (1, 100, 100, 3, 3),
        (2, 100, 100, 3, 3),
    ]

    # 밀어야 할 시간을 설정
    stim_shift = {
        (0, 100, 100, 3, 3): 0,
        (1, 100, 100, 3, 3): 50,
        (2, 100, 100, 3, 3): 100,
    }

    # (0, 100, 100, 3, 3)의 프로파일을 가져오기
    reference_stim = (0, 100, 100, 3, 3)
    reference_data = mean_last_stimuli_sdf.query("stim_type == @reference_stim")
    reference_sdf_values = reference_data["mean_sdf"].values[0]

    # 밀린 프로파일들을 더해 새로운 프로파일을 생성
    new_sdf_values = np.zeros_like(reference_sdf_values)
    sdf_length = len(reference_sdf_values)

    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]

        shift = stim_shift.get(stim, 0)  # 설정된 밀림 시간 또는 0
        shifted_sdf_values = np.zeros_like(mean_sdf_values)

        # 밀림 시간을 반영하여 프로파일을 이동
        if shift > 0:
            shifted_sdf_values[shift:sdf_length] = mean_sdf_values[: sdf_length - shift]
        else:
            shifted_sdf_values = mean_sdf_values

        new_sdf_values += shifted_sdf_values

    # (0, 1, 2, 3, 3)의 프로파일을 가져와 플롯
    original_stim = (0, 1, 2, 5, 5)
    original_data = mean_last_stimuli_sdf.query("stim_type == @original_stim")
    original_sdf_values = original_data["mean_sdf"].values[0]
    plt.step(
        np.arange(0, 30, sdf_dt),
        original_sdf_values,
        where="mid",
        label=f"{original_stim}",
    )

    # 새로운 프로파일 플롯
    plt.step(
        np.arange(0, 30, sdf_dt),
        new_sdf_values,
        where="mid",
        label="New Combined Profile",
    )

    # 범례 설정
    plt.legend()
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_profiles_sum012(55).jpg")

if "SDF_PROFILES_0XX_1XX_2XX" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    target_stim = [
        (0, 100, 100, 3, 3),
        (1, 100, 100, 3, 3),
        (2, 100, 100, 3, 3),
    ]

    # 밀어야 할 시간을 설정
    stim_shift = {
        (0, 100, 100, 3, 3): 0,
        (1, 100, 100, 3, 3): 10,
        (2, 100, 100, 3, 3): 20,
    }

    # (0, 100, 100, 3, 3)의 프로파일을 가져오기
    reference_stim = (0, 100, 100, 3, 3)
    reference_data = mean_last_stimuli_sdf.query("stim_type == @reference_stim")
    reference_sdf_values = reference_data["mean_sdf"].values[0]

    # 밀린 프로파일들을 더해 새로운 프로파일을 생성
    new_sdf_values = np.zeros_like(reference_sdf_values)
    sdf_length = len(reference_sdf_values)

    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]

        shift = stim_shift.get(stim, 0)  # 설정된 밀림 시간 또는 0
        shifted_sdf_values = np.zeros_like(mean_sdf_values)

        # 밀림 시간을 반영하여 프로파일을 이동
        if shift > 0:
            shifted_sdf_values[shift:sdf_length] = mean_sdf_values[: sdf_length - shift]
        else:
            shifted_sdf_values = mean_sdf_values

        new_sdf_values += shifted_sdf_values

    # (0, 1, 2, 3, 3)의 프로파일을 가져와 플롯
    original_stim = (0, 1, 2, 1, 1)
    original_data = mean_last_stimuli_sdf.query("stim_type == @original_stim")
    original_sdf_values = original_data["mean_sdf"].values[0]
    plt.step(
        np.arange(0, 30, sdf_dt),
        original_sdf_values,
        where="mid",
        label=f"{original_stim}",
    )

    # 새로운 프로파일 플롯
    plt.step(
        np.arange(0, 30, sdf_dt),
        new_sdf_values,
        where="mid",
        label="New Combined Profile",
    )

    # 범례 설정
    plt.legend()
    # 그리드 라인 추가
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_profiles_sum012(11).jpg")


# %%
# plot correlation time heatmap
# "SDF_CORR_HEATMAP" : plot correlation time heatmap
if "SDF_CORR_HEATMAP" in plot_files:
    df_corr = pd.DataFrame()
    df_err = pd.DataFrame()
    df_err_none = pd.DataFrame()
    for t1 in range(1, 6):
        for t2 in range(1, 6):
            target_stim = (0, 1, 2, t1, t2)
            stand_stim = (0, 1, 2, 3, 3)
            target_data = mean_last_stimuli_sdf.query("stim_type == @target_stim")[
                "mean_sdf"
            ].values[0]
            stand_data = mean_last_stimuli_sdf.query("stim_type == @stand_stim")[
                "mean_sdf"
            ].values[0]
            correlation = np.correlate(target_data, stand_data, mode="full")
            max_corr_index = np.argmax(correlation)
            shift_ms = (max_corr_index - (len(target_data) - 1)) * sdf_dt
            df_corr_ = pd.DataFrame({"t1": [t1], "t2": [t2], "corr_shift": [shift_ms]})
            df_corr = pd.concat([df_corr, df_corr_])

            shift_data = arr_shift(
                target_data, -(max_corr_index - (len(target_data) - 1))
            )
            err = RMSE(shift_data, stand_data)
            df_err_ = pd.DataFrame({"t1": [t1], "t2": [t2], "err": [err]})
            df_err = pd.concat([df_err, df_err_])

            err_none = RMSE(target_data, stand_data)
            df_err_none_ = pd.DataFrame({"t1": [t1], "t2": [t2], "err": [err_none]})
            df_err_none = pd.concat([df_err_none, df_err_none_])
    max_shift_sdf_corr = max(df_corr["corr_shift"])
    min_shift_sdf_corr = min(df_corr["corr_shift"])
    max_error_shift_corr = max(df_err["err"])
    min_error_shift_corr = min(df_err["err"])
    # plot corr heatmap
    fig, axes = plt.subplots()
    fig.set_size_inches(6, 6)
    sns.heatmap(
        df_corr.pivot(index="t1", columns="t2", values="corr_shift"),
        linewidths=0.5,
        annot=True,
        cmap="jet",
        center=0,
        ax=axes,
    )
    axes.set_title("Correlation time shift Heatmap")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap.jpg")
    # plot err heatmap
    fig, axes = plt.subplots()
    fig.set_size_inches(6, 6)
    sns.heatmap(
        df_err.pivot(index="t1", columns="t2", values="err"),
        linewidths=0.5,
        annot=True,
        cmap="jet",
        center=0,
        ax=axes,
    )
    axes.set_title("Error time shift Heatmap")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap_err.jpg")

    data.update(
        {
            f"max_shift_sdf_corr": max_shift_sdf_corr,
            f"min_shift_sdf_corr": min_shift_sdf_corr,
            f"max_error_shift_corr": max_error_shift_corr,
            f"min_error_shift_corr": min_error_shift_corr,
        }
    )
    df_corr.pivot(index="t1", columns="t2", values="corr_shift").to_csv(
        f"{result_subfolder_dir}/sdf_corr_heatmap.csv"
    )
    df_err.pivot(index="t1", columns="t2", values="err").to_csv(
        f"{result_subfolder_dir}/sdf_corr_heatmap_err.csv"
    )
    # plot error none heatmap
    fig, axes = plt.subplots()
    fig.set_size_inches(6, 6)
    sns.heatmap(
        df_err_none.pivot(index="t1", columns="t2", values="err"),
        linewidths=0.5,
        annot=True,
        cmap="jet",
        center=0,
        ax=axes,
    )
    axes.set_title("Error time shift Heatmap")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap_err_none.jpg")

    data_update_tag = True

# %%
# plot correlation time heatmap(normalized)
# "SDF_NORMAL_CORR_HEATMAP" : plot correlation time heatmap(normalized)
if "SDF_NORMAL_CORR_HEATMAP" in plot_files:
    fig, axes = plt.subplots()
    fig.set_size_inches(6, 6)
    df_corr = pd.DataFrame()

    for t1 in range(0, 6):
        for t2 in range(0, 6):
            target_stim = (0, 1, 2, t1, t2)
            stand_stim = (0, 1, 2, 3, 3)
            target_data = mean_last_stimuli_sdf.query("stim_type == @target_stim")[
                "mean_sdf"
            ].values[0]
            stand_data = mean_last_stimuli_sdf.query("stim_type == @stand_stim")[
                "mean_sdf"
            ].values[0]

            # Min-max normalization
            target_data = (target_data - np.min(target_data)) / (
                np.max(target_data) - np.min(target_data)
            )
            stand_data = (stand_data - np.min(stand_data)) / (
                np.max(stand_data) - np.min(stand_data)
            )

            # Calculate correlation
            correlation = np.correlate(target_data, stand_data, mode="full")
            max_corr_index = np.argmax(correlation)
            shift_ms = (max_corr_index - (len(target_data) - 1)) * sdf_dt

            df_corr_ = pd.DataFrame({"t1": [t1], "t2": [t2], "corr_shift": [shift_ms]})
            df_corr = pd.concat([df_corr, df_corr_])

    sns.heatmap(
        df_corr.pivot(index="t1", columns="t2", values="corr_shift"),
        linewidths=0.5,
        annot=True,
        cmap="jet",
        center=0,
        ax=axes,
    )

    max_shift_sdf_normal_corr = max(df_corr["corr_shift"])
    min_shift_sdf_normal_corr = min(df_corr["corr_shift"])
    axes.set_title("Correlation(normalized) time shift Heatmap")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{result_subfolder_dir}/sdf_normal_corr_heatmap.jpg")

    data.update(
        {
            f"max_shift_sdf_normal_corr": max_shift_sdf_normal_corr,
            f"min_shift_sdr_normal_corr": min_shift_sdf_normal_corr,
        }
    )
    data_update_tag = True

# %%
# import indegree, outdegree data
# "IN_OUT_DEGREE" : plot indegree, outdegree plot
if (
    "IN_OUT_DEGREE" in plot_files
    or "IN_OUT_SCATTER" in plot_files
    or "IN_OUT_DENSITY" in plot_files
    or "IN_OUT_DENSITY_KDE" in plot_files
    or "IN_OUT_DENSITY_KL_DIV" in plot_files
):
    g_e_e = np.load("%s/ee_weight_in_%ds_data.npy" % (data_record_dir, 3600))
    row_e_e = np.load("%s/ee_weight_in_%ds_row.npy" % (data_record_dir, 3600))
    col_e_e = np.load("%s/ee_weight_in_%ds_col.npy" % (data_record_dir, 3600))
    ee_dense = sp.sparse.csr_matrix((g_e_e, (row_e_e, col_e_e))).todense()
    indegree, outdegree = weight_in_out_plot(ee_dense)
    indegree = np.ravel(indegree)
    outdegree = np.ravel(outdegree)
    indegree_s0 = indegree[input_sets[0][np.where(input_sets[0] < 800)]]
    outdegree_s0 = outdegree[input_sets[0][np.where(input_sets[0] < 800)]]
    indegree_s1 = indegree[input_sets[1][np.where(input_sets[1] < 800)]]
    outdegree_s1 = outdegree[input_sets[1][np.where(input_sets[1] < 800)]]
    indegree_s2 = indegree[input_sets[2][np.where(input_sets[2] < 800)]]
    outdegree_s2 = outdegree[input_sets[2][np.where(input_sets[2] < 800)]]

# %%
# export indegree-outdegree data with .csv file
if "IN_OUT_DEGREE_CSV" in plot_files:
    g_e_e = np.load("%s/ee_weight_in_%ds_data.npy" % (data_record_dir, 0))
    row_e_e = np.load("%s/ee_weight_in_%ds_row.npy" % (data_record_dir, 0))
    col_e_e = np.load("%s/ee_weight_in_%ds_col.npy" % (data_record_dir, 0))
    ee_dense = sp.sparse.csr_matrix((g_e_e, (row_e_e, col_e_e))).todense()
    indegree, outdegree = weight_in_out_plot(ee_dense)
    indegree = np.ravel(indegree)
    outdegree = np.ravel(outdegree)
    indegree_s0 = indegree[input_sets[0][np.where(input_sets[0] < 800)]]
    outdegree_s0 = outdegree[input_sets[0][np.where(input_sets[0] < 800)]]
    indegree_s1 = indegree[input_sets[1][np.where(input_sets[1] < 800)]]
    outdegree_s1 = outdegree[input_sets[1][np.where(input_sets[1] < 800)]]
    indegree_s2 = indegree[input_sets[2][np.where(input_sets[2] < 800)]]
    outdegree_s2 = outdegree[input_sets[2][np.where(input_sets[2] < 800)]]

    # export indegree-outdegree data with .csv file
    in_out_degree = pd.DataFrame(
        {
            "indegree": indegree,
            "outdegree": outdegree,
        }
    )
    in_out_degree.to_csv(f"{result_subfolder_dir}/in_out_degree_0s.csv", index=False)
    s0_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s0,
            "outdegree": outdegree_s0,
        }
    )
    s0_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s0_in_out_degree_0s.csv", index=False
    )
    s1_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s1,
            "outdegree": outdegree_s1,
        }
    )
    s1_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s1_in_out_degree_0s.csv", index=False
    )
    s2_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s2,
            "outdegree": outdegree_s2,
        }
    )
    s2_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s2_in_out_degree_0s.csv", index=False
    )

    g_e_e = np.load("%s/ee_weight_in_%ds_data.npy" % (data_record_dir, 3600))
    row_e_e = np.load("%s/ee_weight_in_%ds_row.npy" % (data_record_dir, 3600))
    col_e_e = np.load("%s/ee_weight_in_%ds_col.npy" % (data_record_dir, 3600))
    ee_dense = sp.sparse.csr_matrix((g_e_e, (row_e_e, col_e_e))).todense()
    indegree, outdegree = weight_in_out_plot(ee_dense)
    indegree = np.ravel(indegree)
    outdegree = np.ravel(outdegree)
    indegree_s0 = indegree[input_sets[0][np.where(input_sets[0] < 800)]]
    outdegree_s0 = outdegree[input_sets[0][np.where(input_sets[0] < 800)]]
    indegree_s1 = indegree[input_sets[1][np.where(input_sets[1] < 800)]]
    outdegree_s1 = outdegree[input_sets[1][np.where(input_sets[1] < 800)]]
    indegree_s2 = indegree[input_sets[2][np.where(input_sets[2] < 800)]]
    outdegree_s2 = outdegree[input_sets[2][np.where(input_sets[2] < 800)]]

    # export indegree-outdegree data with .csv file
    in_out_degree = pd.DataFrame(
        {
            "indegree": indegree,
            "outdegree": outdegree,
        }
    )
    in_out_degree.to_csv(f"{result_subfolder_dir}/in_out_degree_3600s.csv", index=False)
    s0_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s0,
            "outdegree": outdegree_s0,
        }
    )
    s0_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s0_in_out_degree_3600s.csv", index=False
    )
    s1_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s1,
            "outdegree": outdegree_s1,
        }
    )
    s1_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s1_in_out_degree_3600s.csv", index=False
    )
    s2_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s2,
            "outdegree": outdegree_s2,
        }
    )
    s2_in_out_degree.to_csv(
        f"{result_subfolder_dir}/s2_in_out_degree_3600s.csv", index=False
    )

# %%
# plot indegree, outdegree scatter plot
# "IN_OUT_SCATTER" : plot indegree, outdegree scatter plot
if "IN_OUT_SCATTER" in plot_files:
    fig, axes = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.scatter(indegree, outdegree, s=5, color="gray", label="others")
    plt.scatter(indegree_s2, outdegree_s2, s=5, color="lime", label="s2")
    plt.scatter(indegree_s1, outdegree_s1, s=5, color="blue", label="s1")
    plt.scatter(indegree_s0, outdegree_s0, s=5, color="red", label="s0")
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.xlabel("indegree")
    plt.ylabel("outdegree")
    plt.title("indegree outdegree weight sum at 3600 s")
    plt.legend()
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig("%s/in_out_scatter_plot.png" % (result_subfolder_dir))
# %%
# plot density plot
# "IN_OUT_DENSITY" : plot indegree, outdegree density plot
if "IN_OUT_DENSITY" in plot_files:
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.xlabel("indegree")
    plt.ylabel("outdegree")
    sns.kdeplot(
        x=indegree_s2,
        y=outdegree_s2,
        fill=True,
        cmap="Greens",
        alpha=0.7,
        ax=ax,
        levels=5,
    )
    sns.kdeplot(
        x=indegree_s1,
        y=outdegree_s1,
        fill=True,
        cmap="Blues",
        alpha=0.7,
        ax=ax,
        levels=5,
    )
    sns.kdeplot(
        x=indegree_s0,
        y=outdegree_s0,
        fill=True,
        cmap="Reds",
        alpha=0.7,
        ax=ax,
        levels=5,
    )
    plt.legend(["s0", "s1", "s2"])
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    fig.savefig(f"{result_subfolder_dir}/in_out_density_plot.jpg")

# %%
# Kernel Density Estimation
if "IN_OUT_DENSITY_KDE" in plot_files:
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    from scipy.stats import gaussian_kde

    x1 = indegree_s0
    y1 = outdegree_s0
    x2 = indegree_s1
    y2 = outdegree_s1
    x3 = indegree_s2
    y3 = outdegree_s2

    data1 = np.vstack((x1, y1))
    data2 = np.vstack((x2, y2))
    data3 = np.vstack((x3, y3))
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    kde3 = gaussian_kde(data3)

    # Generate grid for plotting density estimates
    xmin, xmax = 0, 400
    ymin, ymax = 0, 400
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Calculate density
    Z0 = np.reshape(kde1(positions).T, X.shape).T
    Z1 = np.reshape(kde2(positions).T, X.shape).T
    Z2 = np.reshape(kde3(positions).T, X.shape).T

    # Calculate the mean of the product of the densities (corr)
    s0s1 = np.sum(Z0 * Z1)
    s1s2 = np.sum(Z1 * Z2)
    s2s0 = np.sum(Z2 * Z0)

    np.savetxt(f"{result_subfolder_dir}/s0_density.csv", Z0, delimiter=",")
    np.savetxt(f"{result_subfolder_dir}/s1_density.csv", Z1, delimiter=",")
    np.savetxt(f"{result_subfolder_dir}/s2_density.csv", Z2, delimiter=",")

    # Maximum density value for normalization
    max_density = max(Z0.max(), Z1.max(), Z2.max())

    # Create RGBA arrays
    rgba0 = np.zeros((Z0.shape[0], Z0.shape[1], 4))
    rgba0[..., 0] = 1  # Red
    rgba0[..., 1] = 0  # Green
    rgba0[..., 2] = 0  # Blue
    rgba0[..., 3] = Z0 / max_density  # Alpha

    rgba1 = np.zeros((Z1.shape[0], Z1.shape[1], 4))
    rgba1[..., 0] = 0  # Red
    rgba1[..., 1] = 0  # Green
    rgba1[..., 2] = 1  # Blue
    rgba1[..., 3] = Z1 / max_density  # Alpha

    rgba2 = np.zeros((Z2.shape[0], Z2.shape[1], 4))
    rgba2[..., 0] = 0  # Red
    rgba2[..., 1] = 1  # Green
    rgba2[..., 2] = 0  # Blue
    rgba2[..., 3] = Z2 / max_density  # Alpha

    # Visualize the results
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.xlabel("indegree")
    plt.ylabel("outdegree")
    plt.imshow(rgba2, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
    plt.imshow(rgba1, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
    plt.imshow(rgba0, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
    plt.scatter(x1, y1, c="red", s=10)
    plt.scatter(x2, y2, c="blue", s=10)
    plt.scatter(x3, y3, c="green", s=10)

    fig.savefig(f"{result_subfolder_dir}/in_out_density_plot(2).jpg")

    data.update(
        {
            "bc_s0-s1": s0s1,
            "bc_s1-s2": s1s2,
            "bc_s2-s0": s2s0,
        }
    )
    data_update_tag = True

# %%
# plot tauPlus distribution
# "TAU_PLUS" : plot tauPlus distribution
if "TAU_PLUS" in plot_files:
    tp_e_e = np.load("%s/tauPlus_data.npy" % (data_record_dir))
    mean = np.mean(tp_e_e)
    tp_sd = np.std(tp_e_e)
    tp_df = pd.DataFrame(tp_e_e)
    tp_df.to_csv("%s/tauPlus_data.csv" % (result_subfolder_dir))
    fig, ax = plt.subplots()
    plt.hist(tp_e_e, bins=200)
    plt.title(
        "tauPlus distribution at\n t-a:%f t-b:%f\n mean: %f, sd: %f"
        % (
            params["tauPlus_a"],
            params["tauPlus_b"],
            mean,
            tp_sd,
        )
    )
    plt.savefig("%s/tauPlus_distribution.png" % (result_subfolder_dir))

    data.update(
        {
            "tp_sd": float(tp_sd),
        }
    )
    data_update_tag = True

# %%
# plot tauMinus distribution
# "TAU_MINUS" : plot tauMinus distribution
if "TAU_MINUS" in plot_files:
    tm_e_e = np.load("%s/tauMinus_data.npy" % (data_record_dir))
    mean = np.mean(tm_e_e)
    tm_sd = np.std(tm_e_e)
    tm_df = pd.DataFrame(tm_e_e)
    tm_df.to_csv("%s/tauMinus_data.csv" % (result_subfolder_dir))
    fig, ax = plt.subplots()
    plt.hist(tm_e_e, bins=200)
    plt.title(
        "tauMinus distribution at\n t-a:%f t-b:%f\n mean: %f, sd: %f"
        % (
            params["tauMinus_a"],
            params["tauMinus_b"],
            mean,
            tm_sd,
        )
    )
    plt.savefig("%s/tauMinus_distribution.png" % (result_subfolder_dir))

    data.update(
        {
            "tm_sd": float(tm_sd),
        }
    )
    data_update_tag = True

# %%
# export sdf_indicator.json
if data_update_tag:
    with open(indicator_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
# %%
