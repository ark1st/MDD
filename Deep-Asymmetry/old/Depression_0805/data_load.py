#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import os

#%%
file_path = 'C:/python/depression_dataset/preprocessing/PCA/MDD/'
H_dataset_Filenames = os.listdir(file_path)
MDD_dataset_Filenames = os.listdir(file_path)
all_df = []
y=[]

for k in range(len(H_dataset_Filenames)):
    print('loading dataset : ',k)
    data_first = pd.read_csv(file_path+H_dataset_Filenames[k])
    all_df.append(data_first)
    y.append(0)

for k in range(len(H_dataset_Filenames)):
    print('loading dataset : ',k)
    data_first = pd.read_csv(file_path+MDD_dataset_Filenames[k])
    all_df.append(data_first)
    y.append(1)

print(all_df, y)    
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_df, y, test_size=0.2, random_state = 1)

# %%
X_train_H  = []
X_train_MDD  =  []
y_train_H  = []
y_train_MDD  =  []

for i in range(len(X_train)):
    if y_train[i]  ==  0:
        X_train_H.append(X_train[i])
    else:
        X_train_MDD.append(X_train[i])
# %%


# %%
