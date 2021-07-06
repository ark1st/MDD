import numpy as np
import matplotlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sn

# 뇌파 그림그리기 형식

row_list = [[]]
print(row_list)
image = np.array(row_list)
print('row_list: ', row_list)

fig, axis = plt.subplots()
height, width = image.shape
# im = axis.imshow(image, cmap='seismic', vmin=-1, vmax=1)

df_cm = pd.DataFrame(image)
plt.figure(figsize=(8, 3))
sn.set(font_scale=1)
sn.heatmap(df_cm,cmap='jet', annot=True, vmin=-1, vmax=1)


plt.axis('off'), plt.xticks([]), plt.yticks([])
#axis.axes.get_xaxis().set_visible(False)
#axis.axes.get_yaxis().set_visible(False)
#plt.tight_layout()
#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

plt.show()

'''
#이미지 보여주기
    fig.colorbar(im, shrink=0.5, aspect=5)
    ax.set_xlabel('H_EC')
    plt.show()
'''