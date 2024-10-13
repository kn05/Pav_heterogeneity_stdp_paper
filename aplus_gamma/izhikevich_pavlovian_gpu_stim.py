import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

from config import *

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from time import perf_counter

from common import (
    izhikevich_dopamine_model,
    izhikevich_stdp_model,
    build_model,
    get_params,
    convert_spikes,
)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_start_end_stim(stim_counts):
    end_stimuli = np.cumsum(stim_counts)
    start_stimuli = np.empty_like(end_stimuli)
    start_stimuli[0] = 0
    start_stimuli[1:] = end_stimuli[0:-1]

    return start_stimuli, end_stimuli


def save_pop_data(pop, pop_name, model_t, params, data_record_dir):
    # Pull variables and connectivity from the device

    if pop_name == "ee" or pop_name == "ei":
        pop.pull_var_from_device("g")
        pop.pull_connectivity_from_device()
        # Get variable values
        g_values = pop.get_var_values("g")
        row_values = np.array(pop.get_sparse_pre_inds())
        col_values = np.array(pop.get_sparse_post_inds())

    elif pop_name == "ie" or pop_name == "ii":
        pop.pull_connectivity_from_device()
        g_values = params["inh_weight"]
        row_values = np.array(pop.get_sparse_pre_inds())
        col_values = np.array(pop.get_sparse_post_inds())

    # Determine the time interval for file naming
    time_interval = int(
        (model_t // (params["record_time_sec"] * 1000)) * params["record_time_sec"]
    )

    # Save the data
    np.save(f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_data", g_values)
    np.save(f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_row", row_values)
    np.save(f"{data_record_dir}/{pop_name}_weight_in_{time_interval}s_col", col_values)


# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
stim_noise_model = genn_model.create_custom_current_source_class(
    "stim_noise",
    param_names=["n", "stimMagnitude"],
    var_name_types=[
        ("startStim", "unsigned int"),
        ("endStim", "unsigned int", VarAccess_READ_ONLY),
    ],
    extra_global_params=[("stimTimes", "scalar*")],
    injection_code="""
        scalar current = ($(gennrand_uniform) * $(n) * 2.0) - $(n);
        if($(startStim) != $(endStim) && $(t) >= $(stimTimes)[$(startStim)]) {
           current += $(stimMagnitude);
           $(startStim)++;
        }
        $(injectCurrent, current);
        """,
)

# Load json data from params.json
with open(os.path.join(data_subfolder_dir, "params.json")) as params_file:
    params = json.load(params_file)

# Set random seed
if "seed" in params:
    np.random.seed(params["seed"])

# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
stim_gen_start_time = perf_counter()
input_sets = [
    np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
    for _ in range(params["num_stimuli_sets"])
]
input_sets.append(np.array([]))

# Load target_stimuli_set and stimuli_sets_intervals
target_stimuli_set = np.loadtxt(
    os.path.join(data_subfolder_dir, "target_stimuli_set.txt"), dtype=int
)
stimuli_sets_intervals = np.loadtxt(
    os.path.join(data_subfolder_dir, "stimuli_sets_intervals.txt"), dtype=int
)

# Lists of stimulus and reward times for use when plotting
start_stimulus_times = []
end_stimulus_times = []
start_reward_times = []
end_reward_times = []

# Create list for each neuron
neuron_stimuli_times = [[] for _ in range(num_cells)]
total_num_exc_stimuli = 0
total_num_inh_stimuli = 0

# Create zeroes numpy array to hold reward timestep bitmask
reward_timesteps = np.zeros((params["duration_timestep"] + 31) // 32, dtype=np.uint32)

# Loop while stimuli are within simulation duration
next_stimuli_timestep = np.random.randint(
    params["min_inter_stimuli_interval_timestep"],
    params["max_inter_stimuli_interval_timestep"],
)
cycle_num = 0
while next_stimuli_timestep < params["duration_timestep"]:
    # Pick a stimuli set to present at this timestep
    if cycle_num != 0 and cycle_num % 100 == 0:
        stimuli_set = target_stimuli_set
    else:
        first_part = np.random.choice(101, 3, replace=True)
        second_part = np.random.choice(20, 2, replace=True)
        result = np.concatenate((first_part, second_part))
        stimuli_set = result

    if next_stimuli_timestep > (
        params["duration_timestep"] - params["record_time_timestep"]
    ):
        if cycle_num != 0 and cycle_num % 100 == 0:
            stimuli_set = target_stimuli_set
        else:
            stimuli_set = stimuli_sets_intervals[
                np.random.randint(0, len(stimuli_sets_intervals))
            ]

    # Loop through neurons in stimuli set and add time to list
    for n in input_sets[stimuli_set[0]]:
        neuron_stimuli_times[n].append(next_stimuli_timestep * params["timestep_ms"])
    for n in input_sets[stimuli_set[1]]:
        neuron_stimuli_times[n].append(
            (next_stimuli_timestep + stimuli_set[3]) * params["timestep_ms"]
        )
    for n in input_sets[stimuli_set[2]]:
        neuron_stimuli_times[n].append(
            (next_stimuli_timestep + stimuli_set[3] + stimuli_set[4])
            * params["timestep_ms"]
        )

    # Count the number of excitatory neurons in input set and add to total
    num_exc_in_input_set = np.sum(input_sets[stimuli_set[0]] < params["num_excitatory"])
    total_num_exc_stimuli += num_exc_in_input_set
    total_num_inh_stimuli += num_cells - num_exc_in_input_set

    # If we should be recording at this point, add stimuli to list
    if next_stimuli_timestep < params["record_time_timestep"]:
        start_stimulus_times.append(
            (
                next_stimuli_timestep * params["timestep_ms"],
                stimuli_set[0],
                stimuli_set[1],
                stimuli_set[2],
                stimuli_set[3],
                stimuli_set[4],
            )
        )
    elif next_stimuli_timestep > (
        params["duration_timestep"] - params["record_time_timestep"]
    ):
        end_stimulus_times.append(
            (
                next_stimuli_timestep * params["timestep_ms"],
                stimuli_set[0],
                stimuli_set[1],
                stimuli_set[2],
                stimuli_set[3],
                stimuli_set[4],
            )
        )

    # If this is the rewarded stimuli
    if (stimuli_set == target_stimuli_set).all() and next_stimuli_timestep <= (
        params["duration_timestep"] - params["record_time_timestep"]
    ):
        # Determine time of next reward
        reward_timestep = next_stimuli_timestep + np.random.randint(
            params["max_reward_delay_timestep"]
        )

        # If this is within simulation
        if reward_timestep < params["duration_timestep"]:
            # Set bit in reward timesteps bitmask
            reward_timesteps[reward_timestep // 32] |= 1 << (reward_timestep % 32)

            # If we should be recording at this point, add reward to list
            if reward_timestep < params["record_time_timestep"]:
                start_reward_times.append(reward_timestep * params["timestep_ms"])
            elif reward_timestep > (
                params["duration_timestep"] - params["record_time_timestep"]
            ):
                end_reward_times.append(reward_timestep * params["timestep_ms"])

    # Advance to next stimuli
    next_stimuli_timestep += np.random.randint(
        params["min_inter_stimuli_interval_timestep"],
        params["max_inter_stimuli_interval_timestep"],
    )
    cycle_num += 1

# Count stimuli each neuron should emit
neuron_stimuli_counts = [len(n) for n in neuron_stimuli_times]

stim_gen_end_time = perf_counter()
print(
    "Stimulus generation time: %fms"
    % ((stim_gen_end_time - stim_gen_start_time) * 1000.0)
)

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
# Assert that duration is a multiple of record time
assert (params["duration_timestep"] % params["record_time_timestep"]) == 0

# Build base model
model, e_pop, i_pop, e_e_pop, e_i_pop, i_e_pop, i_i_pop = build_model(
    "izhikevich_pavlovian_gpu_stim", params, reward_timesteps
)

# Current source parameters
curr_source_params = {"n": 6.5, "stimMagnitude": params["stimuli_current"]}

# Calculate start and end indices of stimuli to be injected by each current source
start_exc_stimuli, end_exc_stimuli = get_start_end_stim(
    neuron_stimuli_counts[: params["num_excitatory"]]
)
start_inh_stimuli, end_inh_stimuli = get_start_end_stim(
    neuron_stimuli_counts[params["num_excitatory"] :]
)

# Current source initial state
exc_curr_source_init = {"startStim": start_exc_stimuli, "endStim": end_exc_stimuli}
inh_curr_source_init = {"startStim": start_inh_stimuli, "endStim": end_inh_stimuli}

# Add background current sources
e_curr_pop = model.add_current_source(
    "ECurr", stim_noise_model, "E", curr_source_params, exc_curr_source_init
)
i_curr_pop = model.add_current_source(
    "ICurr", stim_noise_model, "I", curr_source_params, inh_curr_source_init
)

# Set stimuli times
e_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[: params["num_excitatory"]])
)
i_curr_pop.set_extra_global_param(
    "stimTimes", np.hstack(neuron_stimuli_times[params["num_excitatory"] :])
)

if params["build_model"]:
    print("Building model")
    model.build()

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
# Load model, allocating enough memory for recording
print("Loading model")
model.load(num_recording_timesteps=params["record_time_timestep"])

print("Simulating")
# Loop through timesteps
sim_start_time = perf_counter()
start_exc_spikes = None if params["use_genn_recording"] else []
start_inh_spikes = None if params["use_genn_recording"] else []
end_exc_spikes = None if params["use_genn_recording"] else []
end_inh_spikes = None if params["use_genn_recording"] else []

# to save 0 sec files
save_pop_data(e_e_pop, "ee", model.t, params, data_record_dir)
save_pop_data(e_i_pop, "ei", model.t, params, data_record_dir)
save_pop_data(i_e_pop, "ie", model.t, params, data_record_dir)
save_pop_data(i_i_pop, "ii", model.t, params, data_record_dir)

e_e_pop.pull_var_from_device("tauPlus")
e_e_pop.pull_var_from_device("tauMinus")
e_e_pop.pull_var_from_device("aPlus")
e_e_pop.pull_var_from_device("aMinus")
tauplus_e_e = e_e_pop.get_var_values("tauPlus")
tauminus_e_e = e_e_pop.get_var_values("tauMinus")
aplus_e_e = e_e_pop.get_var_values("aPlus")
aminus_e_e = e_e_pop.get_var_values("aMinus")

np.save(
    "%s/tauPlus_data" % (data_record_dir),
    tauplus_e_e,
)
np.save(
    "%s/tauMinus_data" % (data_record_dir),
    tauminus_e_e,
)
np.save(
    "%s/aPlus_data" % (data_record_dir),
    aplus_e_e,
)
np.save(
    "%s/aMinus_data" % (data_record_dir),
    aminus_e_e,
)

while model.t < params["duration_ms"]:
    # Simulation
    model.step_time()

    if params["use_weight_record"] == True:
        if model.t % (params["record_time_sec"] * 1000) == 0:
            save_pop_data(e_e_pop, "ee", model.t, params, data_record_dir)
            save_pop_data(e_i_pop, "ei", model.t, params, data_record_dir)

    if params["use_genn_recording"]:
        # If we've just finished simulating the initial recording interval
        if model.timestep == params["record_time_timestep"]:
            # Download recording data
            model.pull_recording_buffers_from_device()

            start_exc_spikes = e_pop.spike_recording_data
            start_inh_spikes = i_pop.spike_recording_data
        # Otherwise, if we've finished entire simulation
        elif model.timestep == params["duration_timestep"]:
            # Download recording data
            model.pull_recording_buffers_from_device()

            end_exc_spikes = e_pop.spike_recording_data
            end_inh_spikes = i_pop.spike_recording_data
    else:
        if model.timestep <= params["record_time_timestep"]:
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            start_exc_spikes.append(np.copy(e_pop.current_spikes))
            start_inh_spikes.append(np.copy(i_pop.current_spikes))
        elif model.timestep > (
            params["duration_timestep"] - params["record_time_timestep"]
        ):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            end_exc_spikes.append(np.copy(e_pop.current_spikes))
            end_inh_spikes.append(np.copy(i_pop.current_spikes))

sim_end_time = perf_counter()
print("Simulation time: %fms" % ((sim_end_time - sim_start_time) * 1000.0))

if not params["use_genn_recording"]:
    start_timesteps = np.arange(0.0, params["record_time_ms"], params["timestep_ms"])
    end_timesteps = np.arange(
        params["duration_ms"] - params["record_time_ms"],
        params["duration_ms"],
        params["timestep_ms"],
    )

    start_exc_spikes = convert_spikes(start_exc_spikes, start_timesteps)
    start_inh_spikes = convert_spikes(start_inh_spikes, start_timesteps)
    end_exc_spikes = convert_spikes(end_exc_spikes, end_timesteps)
    end_inh_spikes = convert_spikes(end_inh_spikes, end_timesteps)

if params["measure_timing"]:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# plot(
#     start_exc_spikes,
#     start_inh_spikes,
#     end_exc_spikes,
#     end_inh_spikes,
#     start_stimulus_times,
#     start_reward_times,
#     end_stimulus_times,
#     end_reward_times,
#     20000.0,
#     params,
# )

# Save plot
# plt.savefig(
#     f"{params['cwd']}/{params['fig_path']}/{params['dt1']}_{params['dt2']}_plot.jpg")

# Read rewards

with open(
    f"{data_csv_dir}/izhikevich_stimulus_times.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    for stimuls in start_stimulus_times:
        spamwriter.writerow(
            [stimuls[0], stimuls[1], stimuls[2], stimuls[3], stimuls[4], stimuls[5]]
        )
    for stimuls in end_stimulus_times:
        spamwriter.writerow(
            [stimuls[0], stimuls[1], stimuls[2], stimuls[3], stimuls[4], stimuls[5]]
        )

with open(
    f"{data_csv_dir}/izhikevich_reward_times.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    for reward in start_reward_times:
        spamwriter.writerow([reward])
    for reward in end_reward_times:
        spamwriter.writerow([reward])

with open(
    f"{data_csv_dir}/izhikevich_e_spikes.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    spamwriter.writerow(["Time [ms]", " Neuron ID"])
    for time, ids in zip(start_exc_spikes[0], start_exc_spikes[1]):
        spamwriter.writerow([time, ids])
    for time, ids in zip(end_exc_spikes[0], end_exc_spikes[1]):
        spamwriter.writerow([time, ids])

with open(
    f"{data_csv_dir}/izhikevich_i_spikes.csv",
    "w",
    newline="\n",
) as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=",")
    spamwriter.writerow(["Time [ms]", " Neuron ID"])
    for time, ids in zip(start_inh_spikes[0], start_inh_spikes[1]):
        spamwriter.writerow([time, ids])
    for time, ids in zip(end_inh_spikes[0], end_inh_spikes[1]):
        spamwriter.writerow([time, ids])
