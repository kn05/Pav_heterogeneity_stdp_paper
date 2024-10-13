import os
import numpy as np
import subprocess
import time

# GPU 0을 사용하도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

python_path = "/home/kimbell/anaconda3/envs/SNN/bin/python"
script_dir = os.path.dirname(os.path.realpath(__file__))
parameter_generator_path = os.path.join(script_dir, "paramater_generator.py")
stimuli_sets_intervals_generator_path = os.path.join(
    script_dir, "stimuli_sets_intervals_generator.py"
)
simulation_path = os.path.join(script_dir, "izhikevich_pavlovian_gpu_stim.py")
plot_path = os.path.join(script_dir, "plot.py")

start = 0
dt1 = [3] * 21
dt2 = [3] * 21
aPlus_a = [
    1000,
    400,
    100.0,
    25.0,
    11.11111111111111,
    6.25,
    4.0,
    2.7777777777777777,
    2.0408163265306123,
    1.5625,
    1.234567901234568,
    1.0,
    0.8264462809917356,
    0.6944444444444444,
    0.591715976331361,
    0.5102040816326531,
    0.4444444444444444,
    0.390625,
]
aPlus_b = 0.1 / np.array(aPlus_a)
aMinus_a = [0] * 21
aMinus_b = [0] * 21

i = start
for t1, t2, t3, t4, t5, t6 in zip(dt1, dt2, aPlus_a, aPlus_b, aMinus_a, aMinus_b):
    for seed in range(1234, 1234 + 1):
        start_time = time.time()
        print(i, t1, t2, t3, t4)
        subfolder_name_set = f"data{i:02}"
        env = os.environ.copy()
        env["SUBFOLDER_NAME"] = subfolder_name_set
        subprocess.run(
            [
                python_path,
                parameter_generator_path,
                str(seed),
                str(t1),
                str(t2),
                str(t3),
                str(t4),
                str(t5),
                str(t6),
            ],
            env=env,
        )
        subprocess.run([python_path, stimuli_sets_intervals_generator_path], env=env)
        subprocess.run([python_path, simulation_path], env=env)
        end_time = time.time()
        loop_time = end_time - start_time
        print(f"Loop {i} execution time: {loop_time:.2f} seconds")
        i += 1
