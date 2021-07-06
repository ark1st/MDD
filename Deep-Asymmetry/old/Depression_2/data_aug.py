#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import pandas as pd
from sklearn.decomposition import FastICA, PCA

# %%
######change path
path = '../depression/preprocessing/PCA/H/*'
file_list_in = glob.glob(path)
list_im = [file for file in file_list_in]
list_im.sort()
print("file_list:{}".format(list_im))

# %%
for k in range(len(list_im)):
    data_first = pd.read_csv(list_im[k])
    for i in range(30):
        sliced_data = data_first.iloc[i*2560 : (i+1)*2560]

        if(len(sliced_data.columns) == 20):
            sliced_data.to_csv('./sliced_data/H_PCA/'+'H_EC_'+str(k)+"_"+str(i)+'.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                        'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1'])
        elif(len(sliced_data.columns) == 22):
            sliced_data.to_csv('./sliced_data/H_PCA/'+'H_EC_'+str(k)+"_"+str(i)+'.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                        'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1','23A-23R','24A-24R'])
        else:
            print("else error")

# %%
