import numpy as np
import matplotlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("./result/MDD_EC_relative_power_raw_alpha.csv")

print(data.head())

column_list = []
row_list = []


#7
#30

print(len(data))

for l in range(len(data)):
    row_list = []
    result = 0
    # k is left Hemispheric
    # n is right Hemispheric
    for k in range(0, 6):
        column_list = []
        result = 0
        if k < 2 or k == 5:
            for n in range(9, 17):
                result = 0
                a = data.iloc[l, k]
                b = data.iloc[l, n]
                result += ((a - b) / (a + b))
                column_list.append(result)
            row_list.append(column_list)
        else:
            pass
    image = np.array(row_list)
    print('row_list: ', row_list)

    plt.figure(figsize=(64, 96))
    fig, axis = plt.subplots()
    height, width = image.shape
    im = axis.imshow(image, cmap='seismic', vmin=-1, vmax=1)

    # Without margin
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    # fig.colorbar(im, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig('./plot/save/MDD/MDD_EC_' + str(l) + '_theta.png', bbox_inches='tight', pad_inches=0)


    # df_cm = pd.DataFrame(image)
    # plt.figure(figsize=(8, 3))
    # sn.heatmap(df_cm, annot=True, vmin=-10, vmax=10)
    # plt.show()



'''
#이미지 보여주기
    fig.colorbar(im, shrink=0.5, aspect=5)
    ax.set_xlabel('H_EC')
    plt.show()
'''