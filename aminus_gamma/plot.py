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
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import seaborn as sns

from common import (
    get_params,
    convert_spikes,
)
from plot_common import *


# %%
from config import *

# %%
import json

f_params = open(f"{data_subfolder_dir}/params.json", encoding="UTF-8")
params = json.loads(f_params.read())
sigma = float(os.environ.get("SIGMA"))  # params["sigma"] or 1.0
print(f"sigma: {sigma}")

plot_files = os.environ.get("PLOT_FILES")

os.makedirs(result_subfolder_dir, exist_ok=True)

csv_dir = os.path.join(result_subfolder_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)

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
    "HIST_TARGET",
    "SDF_TARGET",
    "SDF_OTHER",
    "SDF_S012",
    "SDF_CORR_HEATMAP",
    "RASTER",
    "RASTER_SORTED",
    "RASTER_CSV",
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
    chunk_duration = 30  # ms
    sdf_offset = 0  # ms

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
    sdf_dt = 0.01
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

    stimuli_data["histo"] = np.zeros(len(stimuli_data), dtype=object)
    stimuli_data["histo"] = stimuli_data["histo"].apply(
        lambda x: np.zeros(chunk_duration)
    )

    stimuli_data["sdf"] = np.zeros(len(stimuli_data), dtype=object)
    stimuli_data["sdf"] = stimuli_data["sdf"].apply(
        lambda x: np.zeros(int(chunk_duration / sdf_dt))
    )

    # fill stimuli_data's sdf column
    histo_indices = np.searchsorted(total_second_hist[0], stimuli_data["time"])
    sdf_indices = np.searchsorted(second_sdf[0], stimuli_data["time"])
    stimuli_data["histo"] = stimuli_data.apply(
        lambda row: total_second_hist[1][
            histo_indices[row.name] : histo_indices[row.name] + chunk_duration
        ],
        axis=1,
    )
    stimuli_data["sdf"] = stimuli_data.apply(
        lambda row: second_sdf[1][
            sdf_indices[row.name]
            - int(sdf_offset / sdf_dt) : sdf_indices[row.name]
            + int(chunk_duration / sdf_dt)
            - int(sdf_offset / sdf_dt)
        ],
        axis=1,
    )

    # divide with first and last
    first_stimuli_data = stimuli_data[stimuli_data["time"] < params["record_time_ms"]]
    last_stimuli_data = stimuli_data[
        stimuli_data["time"] > (params["duration_ms"] - params["record_time_ms"])
    ]

    # calculate mean of last stimuli's SDF
    # sdf의 평균 계산
    mean_last_stimuli_sdf = (
        last_stimuli_data.groupby("stim_type")["sdf"]
        .apply(lambda x: np.mean(x))
        .reset_index(name="mean_sdf")
    )
    # histo의 평균 계산
    mean_last_stimuli_histo = (
        last_stimuli_data.groupby("stim_type")["histo"]
        .apply(lambda x: np.mean(x))
        .reset_index(name="mean_histo")
    )
    # merge sdf and histo
    mean_last_stimuli = pd.merge(
        mean_last_stimuli_sdf, mean_last_stimuli_histo, on="stim_type"
    )

    # save mean_last_stimuli_sdf to csv
    expanded_mean_sdf = pd.DataFrame(
        mean_last_stimuli["mean_sdf"].tolist(),
        index=mean_last_stimuli["stim_type"],
    ).T
    expanded_mean_sdf.to_csv(f"{csv_dir}/mean_last_stimuli_sdf.csv", index=False)

# %%
# plot target_stim's histogram
# "HIST_TARGET" : plot target_stim's histogram
if "HIST_TARGET" in plot_files:
    target_stim = tuple(
        np.loadtxt(
            os.path.join(data_subfolder_dir, "target_stimuli_set.txt"), dtype=int
        )
    )
    fig, axes = plt.subplots()
    fig.suptitle(f"{target_stim}'s hist")
    selected_row = last_stimuli_data.loc[last_stimuli_data["stim_type"] == target_stim]
    series = np.array(selected_row["histo"].tolist())
    for value in selected_row.values:
        plt.step(
            np.arange(0, chunk_duration),
            value[2],
            where="mid",
            color="blue",
            alpha=0.4,
        )
    avg = np.mean(series)
    std_dev = np.std(series, axis=0)
    cv = std_dev / avg

    group_data = mean_last_stimuli_histo.query("stim_type == @target_stim")
    plt.step(
        np.arange(0, chunk_duration),
        group_data["mean_histo"].values[0],
        where="mid",
        color="red",
    )
    fig.savefig(f"{result_subfolder_dir}/target_histo.png")

    data.update(
        {
            f"avg_target": avg,
            f"cv_target": np.mean(cv),
        }
    )
    data_update_tag = True

# %%
# raster plot
# "RASTER" : plot raster plot
if "RASTER" in plot_files:
    target_stims = {
        (0, 1, 2, params["dt1"], params["dt2"]): "s012",
        (0, 100, 100, params["dt1"], params["dt2"]): "s0",
        (1, 100, 100, params["dt1"], params["dt2"]): "s1",
        (2, 100, 100, params["dt1"], params["dt2"]): "s2",
    }
    for stim, name in target_stims.items():
        selected_row = last_stimuli_data.query("stim_type==@stim")
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("Raster plot")
        start_time = selected_row["time"].values[0]  # ms
        raster_time = 50
        end_time = start_time + raster_time

        # 데이터 준비
        e_mask = (e_spikes["time"] >= start_time) & (e_spikes["time"] < end_time)
        i_mask = (i_spikes["time"] >= start_time) & (i_spikes["time"] < end_time)
        e_plot_data = e_spikes[e_mask]
        i_plot_data = i_spikes[i_mask]

        # 색상 지정 함수
        def get_color(neuron_id):
            if neuron_id in input_sets[0]:
                return "red"
            elif neuron_id in input_sets[1]:
                return "blue"
            elif neuron_id in input_sets[2]:
                return "limegreen"
            else:
                return "gray"

        # 색상 배열 생성
        e_colors = np.array([get_color(id) for id in e_plot_data["id"]])
        i_colors = np.array([get_color(id) for id in i_plot_data["id"]])

        # Excitatory 뉴런 플로팅
        for color in set(e_colors):
            mask = e_colors == color
            ax.hlines(
                e_plot_data["id"][mask],
                e_plot_data["time"][mask],
                e_plot_data["time"][mask] + 0.5,
                colors=color,
                lw=1,
                rasterized=True,
            )

        # Inhibitory 뉴런 플로팅
        for color in set(i_colors):
            mask = i_colors == color
            ax.hlines(
                i_plot_data["id"][mask] + 800,
                i_plot_data["time"][mask],
                i_plot_data["time"][mask] + 0.5,
                colors=color,
                lw=1,
                rasterized=True,
            )

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(0, 1000)
        original_ticks = np.arange(start_time, end_time, 10)
        new_ticks = np.arange(0, raster_time, 10)
        ax.set_xticks(original_ticks)
        ax.set_xticklabels(new_ticks)
        ax.set_yticks([0, 800], ["Excitatory", "Inhibitory"])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron ID")

        # 범례 추가
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="r", lw=2, label="Input set 0"),
            Line2D([0], [0], color="b", lw=2, label="Input set 1"),
            Line2D([0], [0], color="limegreen", lw=2, label="Input set 2"),
            Line2D([0], [0], color="gray", lw=2, label="Other neurons"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        fig.savefig(
            f"{result_subfolder_dir}/raster_plot_dash_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

# %%
# raster plot sorted by first spike time
# "RASTER_SORTED" : plot raster plot sorted by first spike time
if "RASTER_SORTED" in plot_files:
    target_stims = {
        (0, 1, 2, params["dt1"], params["dt2"]): "s012",
        (0, 100, 100, params["dt1"], params["dt2"]): "s0",
        (1, 100, 100, params["dt1"], params["dt2"]): "s1",
        (2, 100, 100, params["dt1"], params["dt2"]): "s2",
    }
    for stim, name in target_stims.items():
        target_stim = stim
        selected_row = last_stimuli_data.query("stim_type==@target_stim")
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("Raster plot (Sorted by first spike time)")

        start_time = selected_row["time"].values[0]  # ms
        raster_time = 50
        end_time = start_time + raster_time

        # 데이터 준비
        e_mask = (e_spikes["time"] >= start_time) & (e_spikes["time"] < end_time)
        i_mask = (i_spikes["time"] >= start_time) & (i_spikes["time"] < end_time)
        e_plot_data = e_spikes[e_mask]
        i_plot_data = i_spikes[i_mask]

        # 색상 지정 함수
        def get_color(neuron_id):
            if neuron_id in input_sets[0]:
                return "red"
            elif neuron_id in input_sets[1]:
                return "blue"
            elif neuron_id in input_sets[2]:
                return "limegreen"
            else:
                return "gray"

        # 뉴런 ID를 첫 발화 시간으로 정렬
        def sort_neurons_by_first_spike(spike_data):
            unique_ids = np.unique(spike_data["id"])
            first_spikes = np.array(
                [
                    (id, np.min(spike_data["time"][spike_data["id"] == id]))
                    for id in unique_ids
                ]
            )
            sorted_indices = np.argsort(first_spikes[:, 1])
            return first_spikes[sorted_indices, 0].astype(int)

        e_sorted_ids = sort_neurons_by_first_spike(e_plot_data)
        i_sorted_ids = sort_neurons_by_first_spike(i_plot_data)

        # ID를 정렬된 순서로 매핑
        e_id_to_y = {id: i for i, id in enumerate(e_sorted_ids)}
        i_id_to_y = {id: i + len(e_sorted_ids) for i, id in enumerate(i_sorted_ids)}

        # Excitatory 뉴런 플로팅
        for id in e_sorted_ids:
            mask = e_plot_data["id"] == id
            spikes = e_plot_data["time"][mask]
            y = e_id_to_y[id]
            ax.hlines(
                [y] * len(spikes),
                spikes,
                spikes + 0.5,
                colors=get_color(id),
                lw=1,
                rasterized=True,
            )

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(0, len(e_sorted_ids) + len(i_sorted_ids))

        original_ticks = np.arange(start_time, end_time, 1)
        new_ticks = np.arange(0, raster_time, 1)
        ax.set_xticks(original_ticks)
        ax.set_xticklabels(new_ticks)

        ax.set_yticks([0, len(e_sorted_ids)], ["Excitatory", "Inhibitory"])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Sorted Neuron Index")

        # 범례 추가
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="r", lw=2, label="Input set 0"),
            Line2D([0], [0], color="b", lw=2, label="Input set 1"),
            Line2D([0], [0], color="limegreen", lw=2, label="Input set 2"),
            Line2D([0], [0], color="gray", lw=2, label="Other neurons"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        fig.savefig(
            f"{result_subfolder_dir}/raster_plot_sorted_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

# %%
# export masked spike data
# "RASTER_CSV" : export masked spike data
# %%
# export masked spike data
# "RASTER_CSV" : export masked spike data

import pandas as pd
import numpy as np


# 발화 매트릭스 생성 함수
def create_spike_matrix(neuron_ids, spike_data, time_bins, target_neurons):
    spike_matrix = pd.DataFrame(0, index=neuron_ids, columns=time_bins)

    for neuron_id in target_neurons:
        mask = spike_data["id"] == neuron_id
        spikes = spike_data["time"][mask]

        for spike_time in spikes:
            spike_time = int(spike_time)
            if spike_time in spike_matrix.columns:
                spike_matrix.loc[neuron_id, spike_time] = 1

    return spike_matrix


# 데이터 준비
stims = {
    (0, 1, 2, params["dt1"], params["dt2"]): "s012_3_3",
    (0, 1, 2, 1, 1): "s012_1_1",
    (0, 1, 2, 5, 5): "s012_5_5",
}

for stim, s_name in stims.items():
    selected_row = last_stimuli_data.query("stim_type==@stim")
    start_time = selected_row["time"].values[0]  # ms
    end_time = start_time + 30  # ms
    time_bins = np.arange(start_time, end_time)  # ms 단위 시간 축 생성

    # 각 세트별로 뉴러드 ID 가져오기
    all_neurons = np.unique(np.concatenate([e_spikes["id"], i_spikes["id"]]))
    input_set_0 = sorted(input_sets[0])
    input_set_1 = sorted(input_sets[1])
    input_set_2 = sorted(input_sets[2])
    # 객체 뉴러드 모두 포함하\ub도록 조정
    other_neurons = np.setdiff1d(
        all_neurons, np.concatenate([input_set_0, input_set_1, input_set_2])
    )

    # 각 세트에 대해 발화 매트릭스 생성 및 내보내기
    for input_set, name in zip(
        [input_set_0, input_set_1, input_set_2, other_neurons],
        ["s0", "s1", "s2", "others"],
    ):
        e_mask = (e_spikes["time"] >= start_time) & (e_spikes["time"] < end_time)
        i_mask = (i_spikes["time"] >= start_time) & (i_spikes["time"] < end_time)

        e_plot_data = e_spikes[e_mask]
        i_plot_data = i_spikes[i_mask]

        plot_data = np.concatenate([e_plot_data, i_plot_data])

        # 발화 매트릭스에 모두 뉴러드 포함, 해당 input_set의 뉴러드에 대해서만 발화 정보 생성
        spike_matrix = create_spike_matrix(all_neurons, plot_data, time_bins, input_set)

        # CSV 파일로 저장
        spike_matrix.to_csv(
            f"{result_subfolder_dir}/spike_raster_matrix_{s_name}_{name}.csv"
        )


# %%
# plot target_stim's SDF
# "SDF_TARGET" : plot target_stim's SDF
if "SDF_TARGET" in plot_files:
    target_stim = tuple(
        np.loadtxt(
            os.path.join(data_subfolder_dir, "target_stimuli_set.txt"), dtype=int
        )
    )
    fig, axes = plt.subplots()
    fig.suptitle(f"{target_stim}'s hist")
    selected_row = last_stimuli_data[last_stimuli_data["stim_type"] == target_stim]
    series = np.array(selected_row["sdf"].tolist())
    for value in selected_row.values:
        plt.step(
            np.arange(-sdf_offset, chunk_duration - sdf_offset, sdf_dt),
            value[3],
            where="mid",
            color="blue",
            alpha=0.4,
        )
    avg = np.mean(series)
    std_dev = np.std(series, axis=0)
    cv = std_dev / avg

    group_data = mean_last_stimuli_sdf.query("stim_type == @target_stim")
    plt.step(
        np.arange(-sdf_offset, chunk_duration - sdf_offset, sdf_dt),
        group_data["mean_sdf"].values[0],
        where="mid",
        color="red",
    )
    fig.savefig(f"{result_subfolder_dir}/target_sdf.png")

    data.update(
        {
            f"avg_target": avg,
            f"cv_target": np.mean(cv),
        }
    )
    data_update_tag = True

# %%
# plot target sdf profiles
# "SDF_PROFILES" : plot (0,1,2,3,3), (4,5,6,3,3) SDF
if "SDF_OTHER" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")
    target_stim = [
        (0, 1, 2, params["dt1"], params["dt2"]),
        (4, 5, 6, params["dt1"], params["dt2"]),
    ]
    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]
        peak_index = np.argmax(mean_sdf_values)
        peak_value = mean_sdf_values[peak_index]
        peak_ms = peak_index * sdf_dt
        # Plot profile
        plt.step(
            np.arange(-sdf_offset, chunk_duration - sdf_offset, sdf_dt),
            mean_sdf_values,
            where="mid",
            label=f"{stim}",
        )
        # Mark the peak with a point and text
        plt.plot(peak_ms, peak_value, "ro")
        plt.text(peak_ms, peak_value, f"({peak_ms:.1f}, {peak_value:.2f})")
    # Set the legend
    plt.legend()
    # Save the graph
    fig.savefig(f"{result_subfolder_dir}/target_other_sdf.png")

# %%
# plot 0xx 1xx 2xx profile
# "SDF_S012" : plot (0,1,2,0,0), (0,1,2,1,1), (0,1,2,2,2) SDF
if "SDF_S012" in plot_files:
    target_stim = [
        # (0, 1, 2, params["dt1"], params["dt2"]),
        (0, 100, 100, params["dt1"], params["dt2"]),
        (1, 100, 100, params["dt1"], params["dt2"]),
        (2, 100, 100, params["dt1"], params["dt2"]),
    ]
    stim_shift = {}

    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    for stim in target_stim:
        group_data = mean_last_stimuli_sdf.query("stim_type == @stim")
        mean_sdf_values = group_data["mean_sdf"].values[0]

        # x축을 조정하여 프로파일을 플롯
        shift = stim_shift.get(stim, 0)
        plt.step(
            np.arange(shift - sdf_offset, chunk_duration - sdf_offset, sdf_dt),
            mean_sdf_values[: int(chunk_duration / sdf_dt) - int(shift / sdf_dt)],
            where="mid",
            label=f"{stim}",
        )

    # 범례 설정
    plt.legend()
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_s012.png")


if "SDF_S012" in plot_files:
    fig, axes = plt.subplots()
    fig.suptitle("SDF profiles")

    target_stim = [
        (0, 100, 100, params["dt1"], params["dt2"]),
        (1, 100, 100, params["dt1"], params["dt2"]),
        (2, 100, 100, params["dt1"], params["dt2"]),
    ]

    # 밀어야 할 시간을 설정
    stim_shift = {
        (0, 100, 100, params["dt1"], params["dt2"]): 0,
        (1, 100, 100, params["dt1"], params["dt2"]): int(params["dt1"] / sdf_dt),
        (2, 100, 100, params["dt1"], params["dt2"]): int(
            (params["dt1"] + params["dt2"]) / sdf_dt
        ),
    }

    # (0, 100, 100, params["dt1"] , params["dt2"])의 프로파일을 가져오기
    reference_stim = (0, 100, 100, params["dt1"], params["dt2"])
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

    # (0, 1, 2, params["dt1"], params["dt2"])의 프로파일을 가져와 플롯
    original_stim = (0, 1, 2, params["dt1"], params["dt2"])
    original_data = mean_last_stimuli_sdf.query("stim_type == @original_stim")
    original_sdf_values = original_data["mean_sdf"].values[0]
    plt.step(
        np.arange(-sdf_offset, chunk_duration - sdf_offset, sdf_dt),
        original_sdf_values,
        where="mid",
        label=f"{original_stim}",
    )

    # 새로운 프로파일 플롯
    plt.step(
        np.arange(-sdf_offset, chunk_duration - sdf_offset, sdf_dt),
        new_sdf_values,
        where="mid",
        label="New Combined Profile",
    )

    # 범례 설정
    plt.legend()
    plt.grid(True)
    # 그래프 저장
    fig.savefig(f"{result_subfolder_dir}/sdf_s012_sum_w_shift.png")

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
            stand_stim = (0, 1, 2, params["dt1"], params["dt2"])
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
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap.png")
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
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap_err.png")

    data.update(
        {
            f"max_shift_sdf_corr": max_shift_sdf_corr,
            f"min_shift_sdf_corr": min_shift_sdf_corr,
            f"max_error_shift_corr": max_error_shift_corr,
            f"min_error_shift_corr": min_error_shift_corr,
        }
    )
    df_corr.pivot(index="t1", columns="t2", values="corr_shift").to_csv(
        f"{csv_dir}/sdf_corr_heatmap.csv"
    )
    df_err.pivot(index="t1", columns="t2", values="err").to_csv(
        f"{csv_dir}/sdf_corr_heatmap_err.csv"
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
    fig.savefig(f"{result_subfolder_dir}/sdf_corr_heatmap_err_none.png")

    data_update_tag = True


# %%
# import indegree, outdegree data
# "IN_OUT_DEGREE" : plot indegree, outdegree plot
if (
    "IN_OUT_DEGREE" in plot_files
    or "IN_OUT_SCATTER" in plot_files
    or "IN_OUT_DENSITY" in plot_files
    or "IN_OUT_DENSITY_KDE" in plot_files
    or "WEIGHT_DIST" in plot_files
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
    in_out_degree.to_csv(f"{csv_dir}/in_out_degree_0s.csv", index=False)
    s0_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s0,
            "outdegree": outdegree_s0,
        }
    )
    s0_in_out_degree.to_csv(f"{csv_dir}/s0_in_out_degree_0s.csv", index=False)
    s1_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s1,
            "outdegree": outdegree_s1,
        }
    )
    s1_in_out_degree.to_csv(f"{csv_dir}/s1_in_out_degree_0s.csv", index=False)
    s2_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s2,
            "outdegree": outdegree_s2,
        }
    )
    s2_in_out_degree.to_csv(f"{csv_dir}/s2_in_out_degree_0s.csv", index=False)

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
    in_out_degree.to_csv(f"{csv_dir}/in_out_degree_3600s.csv", index=False)
    s0_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s0,
            "outdegree": outdegree_s0,
        }
    )
    s0_in_out_degree.to_csv(f"{csv_dir}/s0_in_out_degree_3600s.csv", index=False)
    s1_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s1,
            "outdegree": outdegree_s1,
        }
    )
    s1_in_out_degree.to_csv(f"{csv_dir}/s1_in_out_degree_3600s.csv", index=False)
    s2_in_out_degree = pd.DataFrame(
        {
            "indegree": indegree_s2,
            "outdegree": outdegree_s2,
        }
    )
    s2_in_out_degree.to_csv(f"{csv_dir}/s2_in_out_degree_3600s.csv", index=False)

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
    fig.savefig(f"{result_subfolder_dir}/in_out_density_plot.png")

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

    np.savetxt(f"{csv_dir}/s0_density.csv", Z0, delimiter=",")
    np.savetxt(f"{csv_dir}/s1_density.csv", Z1, delimiter=",")
    np.savetxt(f"{csv_dir}/s2_density.csv", Z2, delimiter=",")

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

    fig.savefig(f"{result_subfolder_dir}/in_out_density_plot(2).png")

    data.update(
        {
            "bc_s0-s1": s0s1,
            "bc_s1-s2": s1s2,
            "bc_s2-s0": s2s0,
        }
    )
    data_update_tag = True

# %%
# plot weight distribution
# "WEIGHT_DIST" : plot weight distribution
if "WEIGHT_DIST" in plot_files:
    g_e_e = np.load("%s/ee_weight_in_%ds_data.npy" % (data_record_dir, 3600))
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    sns.histplot(g_e_e, bins=100, kde=True, ax=ax)
    ax.set_xlim([0, 4])
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.set_title("Weight distribution at 3600 s")
    fig.savefig(f"{result_subfolder_dir}/weight_dist.png")

# %%
# plot tauPlus distribution
# "TAU_PLUS" : plot tauPlus distribution
if "TAU_PLUS" in plot_files:
    tp_e_e = np.load("%s/tauPlus_data.npy" % (data_record_dir))
    mean = np.mean(tp_e_e)
    tp_sd = np.std(tp_e_e)
    tp_df = pd.DataFrame(tp_e_e)
    tp_df.to_csv("%s/tauPlus_data.csv" % (csv_dir))
    fig, ax = plt.subplots()
    plt.hist(tp_e_e, bins=200)
    plt.title(
        "tauPlus distribution at\n mean: %f, sd: %f"
        % (
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
    tm_df.to_csv("%s/tauMinus_data.csv" % (csv_dir))
    fig, ax = plt.subplots()
    plt.hist(tm_e_e, bins=200)
    plt.title(
        "tauMinus distribution at\n mean: %f, sd: %f"
        % (
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

if "A_PLUS" in plot_files:
    ap_e_e = np.load("%s/aPlus_data.npy" % (data_record_dir))
    mean = np.mean(ap_e_e)
    ap_sd = np.std(ap_e_e)
    ap_df = pd.DataFrame(ap_e_e)
    ap_df.to_csv("%s/aPlus_data.csv" % (csv_dir))
    fig, ax = plt.subplots()
    plt.hist(ap_e_e, bins=200)
    plt.title(
        "aPlus distribution at\n mean: %f, sd: %f"
        % (
            mean,
            ap_sd,
        )
    )
    plt.savefig("%s/aPlus_distribution.png" % (result_subfolder_dir))

    data.update(
        {
            "ap_sd": float(ap_sd),
        }
    )
    data_update_tag = True

if "A_MINUS" in plot_files:
    am_e_e = np.load("%s/aMinus_data.npy" % (data_record_dir))
    mean = np.mean(am_e_e)
    am_sd = np.std(am_e_e)
    am_df = pd.DataFrame(am_e_e)
    am_df.to_csv("%s/aMinus_data.csv" % (csv_dir))
    fig, ax = plt.subplots()
    plt.hist(am_e_e, bins=200)
    plt.title(
        "aMinus distribution at\n mean: %f, sd: %f"
        % (
            mean,
            am_sd,
        )
    )
    plt.savefig("%s/aMinus_distribution.png" % (result_subfolder_dir))

    data.update(
        {
            "am_sd": float(am_sd),
        }
    )
    data_update_tag = True


# %%
def draw_histogram(x_ee_dense, ee_dense, ee_dense_condition, color, label):
    # ee_dense 조건에 맞는 ap_ee_dense 값을 필터링하여 히스토그램을 그림
    x_ee_dense_condition = np.array(
        x_ee_dense[
            (ee_dense_condition - 0.1 <= ee_dense)
            & (ee_dense <= ee_dense_condition + 0.1)
            & (x_ee_dense > 0)
        ]
    ).ravel()

    plt.hist(
        x_ee_dense_condition,
        bins=200,
        alpha=0.7,
        label=f"ee_dense = {ee_dense_condition}",
        color=color,
        orientation="vertical",
    )

    # save to csv
    pd.DataFrame(x_ee_dense_condition).to_csv(f"{csv_dir}/{label}.csv")


if "DENSE_HIST" in plot_files:
    if not "ee_dense" in locals():
        g_e_e = np.load("%s/ee_weight_in_%ds_data.npy" % (data_record_dir, 3600))
        row_e_e = np.load("%s/ee_weight_in_%ds_row.npy" % (data_record_dir, 3600))
        col_e_e = np.load("%s/ee_weight_in_%ds_col.npy" % (data_record_dir, 3600))
        ee_dense = sp.sparse.csr_matrix((g_e_e, (row_e_e, col_e_e))).todense()

    if "TAU_PLUS" in plot_files:
        try:
            tp_e_e = np.load(f"{data_record_dir}/tauPlus_data.npy")
            tp_ee_dense = sp.sparse.csr_matrix((tp_e_e, (row_e_e, col_e_e))).todense()

            # 다양한 ee_dense 값에 대한 히스토그램을 생성
            plt.figure()
            conditions_colors_labels = [
                (tp_ee_dense, ee_dense, 0, "blue", 0),
                (tp_ee_dense, ee_dense, 4, "red", 4),
            ]

            for (
                x_ee_dense,
                ee_dense,
                condition,
                color,
                label,
            ) in conditions_colors_labels:
                draw_histogram(x_ee_dense, ee_dense, condition, color, label)

            # 그래프 설정
            plt.xlabel("tp_ee_dense values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title("tp_ee_dense Histogram by ee_dense Value")

            # 그래프 보여주기 및 저장
            plt.show()
            plt.savefig(f"{result_subfolder_dir}/tp_ee_dense_histogram.jpg")
        except FileNotFoundError:
            pass

    if "TAU_MINUS" in plot_files:
        try:
            tm_e_e = np.load(f"{data_record_dir}/tauMinus_data.npy")
            tm_ee_dense = sp.sparse.csr_matrix((tm_e_e, (row_e_e, col_e_e))).todense()

            # 다양한 ee_dense 값에 대한 히스토그램을 생성
            plt.figure()
            conditions_colors_labels = [
                (tm_ee_dense, ee_dense, 0, "blue", 0),
                (tm_ee_dense, ee_dense, 4, "red", 4),
            ]

            for (
                x_ee_dense,
                ee_dense,
                condition,
                color,
                label,
            ) in conditions_colors_labels:
                draw_histogram(x_ee_dense, ee_dense, condition, color, label)

            # 그래프 설정
            plt.xlabel("tm_ee_dense values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title("tm_ee_dense Histogram by ee_dense Value")

            # 그래프 보여주기 및 저장
            plt.show()
            plt.savefig(f"{result_subfolder_dir}/tm_ee_dense_histogram.jpg")
        except FileNotFoundError:
            pass

    if "A_PLUS" in plot_files:
        try:
            ap_e_e = np.load(f"{data_record_dir}/aPlus_data.npy")
            ap_ee_dense = sp.sparse.csr_matrix((ap_e_e, (row_e_e, col_e_e))).todense()

            # 다양한 ee_dense 값에 대한 히스토그램을 생성
            plt.figure()
            conditions_colors_labels = [
                (ap_ee_dense, ee_dense, 0, "blue", "ap_ee_dense_0"),
                (ap_ee_dense, ee_dense, 4, "red", "ap_ee_dense_4"),
            ]

            for (
                x_ee_dense,
                ee_dense,
                condition,
                color,
                label,
            ) in conditions_colors_labels:
                draw_histogram(x_ee_dense, ee_dense, condition, color, label)

            # 그래프 설정
            plt.xlabel("ap_ee_dense values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title("ap_ee_dense Histogram by ee_dense Value")

            # 그래프 보여주기 및 저장
            plt.show()
            plt.savefig(f"{result_subfolder_dir}/ap_ee_dense_histogram.jpg")
        except FileNotFoundError:
            pass

    if "A_MINUS" in plot_files:
        try:
            am_e_e = np.load(f"{data_record_dir}/aMinus_data.npy")
            am_ee_dense = sp.sparse.csr_matrix((am_e_e, (row_e_e, col_e_e))).todense()

            # 다양한 ee_dense 값에 대한 히스토그램을 생성
            plt.figure()
            conditions_colors_labels = [
                (am_ee_dense, ee_dense, 0, "blue", "am_ee_dense_0"),
                (am_ee_dense, ee_dense, 4, "red", "am_ee_dense_4"),
            ]

            for (
                x_ee_dense,
                ee_dense,
                condition,
                color,
                label,
            ) in conditions_colors_labels:
                draw_histogram(x_ee_dense, ee_dense, condition, color, label)

            # 그래프 설정
            plt.xlabel("am_ee_dense values")
            plt.ylabel("Frequency")
            plt.legend()
            plt.title("am_ee_dense Histogram by ee_dense Value")

            # 그래프 보여주기 및 저장
            plt.show()
            plt.savefig(f"{result_subfolder_dir}/am_ee_dense_histogram.jpg")
        except FileNotFoundError:
            pass


# %%
# export sdf_indicator.json
if data_update_tag:
    with open(indicator_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
# %%
