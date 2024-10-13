import os
import numpy as np
import subprocess
import time

# GPU 1을 사용하도록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

python_path = "/home/kimbell/anaconda3/envs/SNN/bin/python"
script_dir = os.path.dirname(os.path.realpath(__file__))
parameter_generator_path = os.path.join(script_dir, "paramater_generator.py")
simulation_path = os.path.join(script_dir, "izhikevich_pavlovian_gpu_stim.py")
plot_path = os.path.join(script_dir, "plot.py")

start = 0
tauminus_a = [
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
tauminus_b = [
    0.0,
    0.05,
    0.2,
    0.8,
    1.8,
    3.2,
    5.0,
    7.2,
    9.8,
    12.8,
    16.2,
    20.0,
    24.2,
    28.8,
    33.8,
    39.2,
    45.0,
    51.2,
]
length = len(tauminus_a)
tauplus_a = [1000] * length
tauplus_b = [0.0] * length

i = start
for t1, t2, t3, t4 in zip(
    tauplus_a,
    tauplus_b,
    tauminus_a,
    tauminus_b,
):
    for seed in range(1234, 1234 + 10):
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
            ],
            env=env,
        )
        subprocess.run([python_path, simulation_path], env=env)
        end_time = time.time()
        loop_time = end_time - start_time
        print(f"Loop {i} execution time: {loop_time:.2f} seconds")
        i += 1
