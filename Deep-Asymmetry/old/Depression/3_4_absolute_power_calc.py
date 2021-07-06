import numpy as np
import matplotlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# 뇌파의 평균 구하기

data_mdd = pd.read_csv("./result/MDD_EC_relative_power_raw_alpha.csv")
data_control = pd.read_csv("./result/H_EC_relative_power_raw_alpha.csv")

print(data_mdd.head())

mdd_result = []
control_result = []
result_1 = 0
#7
#30

print(len(data_mdd))

for l in range(len(data_mdd)):
    result_1 = 0
    # k is left Hemispheric
    # n is right Hemispheric
    for j in range(0, 19):
        result_1 += data_mdd.iloc[l,j]
    mdd_result.append(result_1)
# print(mdd_result)
print('mdd_person_avg:', sum(mdd_result)/float(len(mdd_result)))

result1 = sum(mdd_result)/float(len(mdd_result))

mdd_result = []
control_result = []
result_2 = 0

for m in range(len(data_control)):
    result_2 = 0
    # k is left Hemispheric
    # n is right Hemispheric
    for k in range(0, 19):
        result_2 += data_control.iloc[m, k]
    control_result.append(result_2)
# print(control_result)
print('controls_person_avg:', sum(control_result)/float(len(control_result)))

result2 = sum(control_result)/float(len(control_result))

print(abs(result2-result1))
