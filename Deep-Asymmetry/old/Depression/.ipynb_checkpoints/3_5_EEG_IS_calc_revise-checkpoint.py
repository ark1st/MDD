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


result = 0
first_row = 0
sum_everything = 0
#7
#30

print(len(data_mdd))

for l in range(len(data_mdd)):
    result = 0
    # k is left Hemispheric
    # n is right Hemispheric
    for m in range(0, 6):
        result = 0
        if m < 2 or m == 5:
            for n in range(9, 17):
                result = 0
                a = data_mdd.iloc[l, m]
                b = data_mdd.iloc[l, n]
                result += ((a - b) / (a + b))
        else:
            pass
    mdd_result.append(result)
print(mdd_result)
print('mdd_person_avg:', sum(mdd_result)/float(len(mdd_result)))

result1 = sum(mdd_result)/float(len(mdd_result))

mdd_result = []
control_result = []
first_person = 0
result = 0


for m in range(len(data_control)):
    first_person = 0
    result = 0
    # k is left Hemispheric
    # n is right Hemispheric
    for k in range(0, 6):
        result = 0
        if k < 2 or k == 5:
            for n in range(9, 17):
                result = 0
                a = data_control.iloc[m, k]
                b = data_control.iloc[m, n]
                result += ((a - b) / (a + b))
        else:
            pass
    control_result.append(result)
print(control_result)
print('controls_person_avg:', sum(control_result)/float(len(control_result)))

result2 = sum(control_result)/float(len(control_result))


print(abs(result1-result2))


'''
    fig, axis = plt.subplots()
    height, width = image.shape
    im = axis.imshow(image,cmap='seismic', vmin=-100, vmax=100)

    #Without margin
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    #fig.colorbar(im, shrink=0.5, aspect=5)
    #plt.show()
    plt.savefig('./plot/save/H/H_EC_'+str(l)+'_alpha.png',bbox_inches='tight', pad_inches=0)
'''

'''
#이미지 보여주기
    fig.colorbar(im, shrink=0.5, aspect=5)
    ax.set_xlabel('H_EC')
    plt.show()
'''