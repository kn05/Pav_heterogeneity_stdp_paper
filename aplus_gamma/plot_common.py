import numpy as np
from collections import Counter
from scipy.signal import find_peaks
from six import iteritems


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
            left = (t - t0 - kwdt) / dt + kwdt / dt
            right = (t - t0 + kwdt) / dt + kwdt / dt
            sdfs[round(left) : round(right)] += x
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
