# code for analyze

# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# %%
env = os.environ
env["SUBFOLDER_NAME"] = "data00"

from config import *


# %%
def weight_in_out_plot(input_numpy_type_peak_t1_t2ay):
    indegree = input_numpy_type_peak_t1_t2ay.sum(axis=0)
    outdegree = input_numpy_type_peak_t1_t2ay.sum(axis=1)
    return indegree, outdegree


def get_excitatory_neurons(input_set, threshold):
    return input_set[input_set < threshold]


def load_pop_data(pop_name, time_interval, data_record_dir):
    # Load data from files
    g_values = np.load(
        f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_data.npy"
    )
    row_values = np.load(
        f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_row.npy"
    )
    col_values = np.load(
        f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_col.npy"
    )

    # Create dense matrix from sparse representation
    if np.size(g_values) == np.size(row_values) == np.size(col_values):
        dense_matrix = sp.sparse.csr_matrix(
            (g_values, (row_values, col_values))
        ).todense()
    elif np.size(g_values) == 1:
        g_values = [g_values] * np.size(row_values)
        dense_matrix = sp.sparse.csr_matrix(
            (g_values, (row_values, col_values))
        ).todense()

    return dense_matrix


# %%
f_params = open(f"{data_subfolder_dir}/params.json", encoding="UTF-8")
params = json.loads(f_params.read())

if "seed" in params:
    np.random.seed(params["seed"])
num_neurons = 1000
# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
input_sets = [
    np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
    for _ in range(params["num_stimuli_sets"])
]

# %%

ee_dense = load_pop_data("ee", 3600, data_record_dir)
ei_dense = load_pop_data("ei", 3600, data_record_dir)
ie_dense = load_pop_data("ie", 0, data_record_dir)
ii_dense = load_pop_data("ii", 0, data_record_dir)

dense = np.zeros((1000, 1000))
num_excitatory = params["num_excitatory"]
dense[:num_excitatory, :num_excitatory] = ee_dense
dense[num_excitatory:, :num_excitatory] = ie_dense
dense[:num_excitatory, num_excitatory:] = ei_dense
dense[num_excitatory:, num_excitatory:] = ii_dense

sns.heatmap(dense, center=0)
plt.show()
# %%
subgroup_excitatory = []
for i, sets in enumerate(input_sets):
    input_neurons = get_excitatory_neurons(sets, num_excitatory)
    subgroup_excitatory.append(input_neurons)
subgroup_excitatory = np.array(subgroup_excitatory)
inhibited_neurons = np.arange(num_excitatory, num_neurons)
flatted_subgroup_excitatory = np.concatenate(subgroup_excitatory)

# %%
for subgroup in subgroup_excitatory:
    s0_to_inh = np.where(
        dense[np.ix_(subgroup_excitatory[0], inhibited_neurons)] > 2, 1, 0
    )
    inh_to_sn = np.where(dense[np.ix_(inhibited_neurons, subgroup)] < 0, 1, 0)
    s0_inh_sn = s0_to_inh @ inh_to_sn
    print(np.sum(s0_inh_sn))

# %%
print(np.sum(np.where(dense[subgroup_excitatory[0], :] > 2, 1, 0)))
print(np.sum(np.where(dense[subgroup_excitatory[1], :] > 2, 1, 0)))
print(np.sum(np.where(dense[subgroup_excitatory[2], :] > 2, 1, 0)))
print(np.sum(np.where(dense[subgroup_excitatory[4], :] > 2, 1, 0)))
# %%
for subgroup in subgroup_excitatory:
    s0_to_sn = np.where(dense[np.ix_(subgroup_excitatory[0], subgroup)] > 2, 1, 0)
    print(np.sum(s0_to_sn))

# %%
print(np.sum(s0_to_inh))

# %%
sns.heatmap(
    dense[subgroup_excitatory[2], :],
    center=0,
    square=True,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
)
plt.show()

# %%
