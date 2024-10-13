import os
import subprocess
import signal
import sys
import readline
import glob
import inquirer

python_path = "/home/kimbell/anaconda3/envs/SNN/bin/python"
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)


# 지정된 디렉토리 내의 파일 이름 목록을 가져오는 함수
def get_code_files(script_dir):
    return [
        os.path.basename(f) for f in glob.glob(os.path.join(script_dir, "*.py"))
    ]  # .py 파일 목록 가져오기


# 파일에서 플롯할 파일 목록을 읽어오는 함수
def get_plot_files_from_file(file_path):
    with open(file_path, "r") as f:
        plot_files = [
            line.strip() for line in f if line.strip()
        ]  # 파일에서 각 줄을 읽어와 공백 제거 후 리스트로 변환
    return plot_files


# 자동완성을 위한 설정 함수
def completer(text, state):
    options = [
        os.path.basename(i) for i in get_code_files(script_dir) if i.startswith(text)
    ]
    if state < len(options):
        return options[state]
    else:
        return None


readline.set_completer(completer)
readline.parse_and_bind("tab: complete")


# 사용자로부터 스크립트 파일 이름 입력 받기
def input_script_file_name():
    code_file_name = input("Enter the code file name: ")
    return code_file_name


code_file_name = input_script_file_name()

# 플롯할 파일 목록을 선택하도록 질문 설정
plot_file_path = os.path.join(
    script_dir, "plot_file.txt"
)  # 플롯 파일 목록이 담긴 텍스트 파일 경로
file_choices = get_plot_files_from_file(plot_file_path)
questions = [
    inquirer.Checkbox(
        "plot_files",
        message="Select the plot files",
        choices=file_choices,
    ),
]

# 질문을 통해 플롯할 파일 목록 받기
answers = inquirer.prompt(questions)
plot_files = ",".join(answers["plot_files"])


def input_sigma():
    sigma_value = input("Enter the sigma value: ")
    return sigma_value


sigma = input_sigma()


# Ctrl+C 신호를 처리하는 핸들러 함수
def signal_handler(sig, frame):
    print("Interrupt received, stopping all processes...")
    sys.exit(0)


# SIGINT 신호를 가로채서 핸들러로 연결
signal.signal(signal.SIGINT, signal_handler)

for i in range(0, 18):
    subfolder_name_set = f"data{i:02}"
    env = os.environ.copy()  # 현재 환경 변수 복사
    print(f"Plotting {subfolder_name_set}...")
    env["SUBFOLDER_NAME"] = subfolder_name_set
    env["PLOT_FILES"] = plot_files  # 플롯할 파일 목록 설정
    env["SIGMA"] = sigma

    plot_path = os.path.join(script_dir, code_file_name)
    # subprocess 모듈을 사용하여 스크립트를 실행
    try:
        subprocess.run([python_path, plot_path], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        break
    print(f"Plotting {subfolder_name_set} completed.")

print("All plotting processes completed.")
