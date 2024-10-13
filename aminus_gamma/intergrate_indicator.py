import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
import os
import json
import readline
import glob

from config import *

# 빈 DataFrame 생성
df = pd.DataFrame()

# 파이썬 실행 파일 경로
python_path = "/home/kimbell/anaconda3/envs/SNN/bin/python"

# 첫 번째 서브폴더에서 params.json 파일을 읽어와 가능한 키를 추출
example_subfolder_name = "data00"
example_json_path = os.path.join(data_dir, example_subfolder_name, "params.json")
with open(example_json_path, "r") as f:
    example_json = json.load(f)
possible_keys = list(example_json.keys())


# 자동완성 기능 설정
def completer(text, state):
    options = [key for key in possible_keys if key.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None


readline.set_completer(completer)
readline.parse_and_bind("tab: complete")

# 사용자로부터 기준이 될 열 입력받기
print(f"사용 가능한 열: {', '.join(possible_keys)}")

# 유효한 입력이 들어올 때까지 반복
while True:
    group_key = input("기준이 될 열을 입력하세요: ")
    if group_key in possible_keys:
        break
    print(f"입력한 열 '{group_key}'은(는) 유효하지 않습니다. 다시 입력해주세요.")

# 120개의 서브폴더에 대해 반복
for i in range(18):
    subfolder_name = f"data{i:02}"
    print(subfolder_name)
    env = os.environ
    env["SUBFOLDER_NAME"] = subfolder_name

    # 각 경로 설정
    data_subfolder_dir = os.path.join(data_dir, subfolder_name)
    data_record_dir = os.path.join(data_subfolder_dir, "record")
    data_csv_dir = os.path.join(data_subfolder_dir, "csv")
    result_subfolder_dir = os.path.join(result_dir, subfolder_name)

    # JSON 파일 경로 설정 및 데이터 로드
    json1_path = os.path.join(data_subfolder_dir, "params.json")
    json2_path = os.path.join(result_subfolder_dir, "sdf_indicator.json")
    with open(json1_path, "r") as f:
        json1 = json.load(f)
    with open(json2_path, "r") as f:
        json2 = json.load(f)

    # JSON 데이터를 결합하여 DataFrame 생성
    js = {"subfolder_name": subfolder_name, **json1, **json2}
    df_ = pd.DataFrame(js, index=[i])
    df = pd.concat([df, df_], ignore_index=True)

# 최종 DataFrame 출력
print(df)

# 데이터프레임 CSV 파일로 저장
df_path = os.path.join(result_dir, "data.csv")
df.to_csv(df_path)

# s0-s1, s1-s2, s2-s0 산점도 생성 및 저장
fig = plt.figure()
ax3 = fig.subplots()
sns.scatterplot(data=df, x="tp_sd", y="bc_s0-s1", label="s0-s1", ax=ax3)
sns.scatterplot(data=df, x="tp_sd", y="bc_s1-s2", label="s1-s2", ax=ax3)
sns.scatterplot(data=df, x="tp_sd", y="bc_s2-s0", label="s2-s0", ax=ax3)
ax3.legend()
overlap_plot_path = os.path.join(result_dir, "overlap.png")
fig.savefig(overlap_plot_path)

# sdf_indicator의 모든 요소에 대해 그룹화하여 평균과 표준편차 계산
indicator_cols = list(json2.keys())
print(indicator_cols)
grouped_mean = df.groupby(group_key)[indicator_cols].mean()
grouped_std = df.groupby(group_key)[indicator_cols].std()

# 새로운 DataFrame 생성
result_df = pd.DataFrame()

# 평균 및 표준편차 데이터 추가
for col in indicator_cols:
    result_df[f"{col}_mean"] = grouped_mean[col]
    result_df[f"{col}_std"] = grouped_std[col]

# 결과 DataFrame 출력 및 CSV 파일로 저장
print(result_df)
result_df_path = os.path.join(result_dir, "sdf_result_data.csv")
result_df.to_csv(result_df_path, index=False)
