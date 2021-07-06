#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import os



# %%
file_path = 'C:/python/depression_dataset/preprocessing/PCA/MDD/'
H_dataset_Filenames = os.listdir(file_path)
MDD_dataset_Filenames = os.listdir(file_path)
all_df = []
y=[]

for k in range(len(H_dataset_Filenames)):
    #print('loading dataset : ',k)
    data_first = pd.read_csv(file_path+H_dataset_Filenames[k])
    all_df.append(data_first)
    y.append(0)

for k in range(len(H_dataset_Filenames)):
    #print('loading dataset : ',k)
    data_first = pd.read_csv(file_path+MDD_dataset_Filenames[k])
    all_df.append(data_first)
    y.append(1)

print(all_df, y)    


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_df, y, test_size=0.2, random_state = 3)

# %%
X_train_H  = []
X_train_MDD  =  []

for i in range(len(X_train)):
    if y_train[i]  ==  0:
        X_train_H.append(X_train[i])
    else:
        X_train_MDD.append(X_train[i])
        X_test_H  = []
 
X_test_H = []        
X_test_MDD = []
for i in range(len(X_test)):
    if y_test[i]  ==  0:
        X_test_H.append(X_test[i])
    else:
        X_test_MDD.append(X_test[i])

del X_train, X_test

# %%
#data_segmentation
seg_X_train_H = []
seg_X_train_MDD = []

for k in range(len(X_train_H)):
    data_first = X_train_H[k]
    for i in range(30):
        sliced_data = data_first.iloc[i*2560 : (i+1)*2560]
        seg_X_train_H.append(sliced_data)

for k in range(len(X_train_MDD)):
    data_first = X_train_MDD[k]
    for i in range(30):
        sliced_data = data_first.iloc[i*2560 : (i+1)*2560]
        seg_X_train_MDD.append(sliced_data)

# %%
from rel_power import relative_Power
#relative_Power(seg_X_train_H, 'relative_power/train_H_PCA_alpha.csv')
#relative_Power(seg_X_train_MDD, 'relative_power/train_MDD_PCA_alpha.csv')
relative_Power(X_test_H, 'relative_power/test_H_PCA_alpha.csv')
#relative_Power(X_test_MDD, 'relative_power/test_MDD_PCA_alpha.csv')


# %%
import draw

#draw.EEG_drawing('relative_power/train_H_PCA.csv','fig/PCA_alpha/train/H/')
#draw.EEG_drawing('relative_power/train_H_PCA_alpha.csv','fig/PCA_alpha/train/MDD/')
draw.EEG_drawing('relative_power/test_H_PCA_alpha.csv','fig/PCA_alpha/test/H/')
#draw.EEG_drawing('relative_power/test_MDD_PCA.csv','fig/PCA_alpha/test/MDD/')

# %%
