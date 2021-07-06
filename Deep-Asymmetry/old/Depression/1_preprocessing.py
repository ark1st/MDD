import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import pandas as pd

from sklearn.decomposition import FastICA, PCA



######change path
path = 'data/H/EC/*'
file_list_in = glob.glob(path)
list_im = [file for file in file_list_in]
list_im.sort()
print("file_list:{}".format(list_im))

for k in range(len(list_im)):
    total_list = []
    print('loading dataset : ', k)
    data_first = pd.read_csv(list_im[k])

    X = data_first.values

    #Nomalization
    #X /= X.std(axis=0)

    # Compute ICA
    ica = FastICA()
    S_ = ica.fit_transform(X)  # Reconstruct signals

    # For comparison, compute PCA
    pca = PCA()
    H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components


    #X�뒗  �븳 Raw �뜲�씠�꽣
    #S_ �뒗 ica
    #H �뒗 pca
    print('X : ',X)
    print('S_:', S_)
    print('H :', H)

    data_frame = pd.DataFrame(S_)

    if(len(data_frame.columns) == 20):
        data_frame.to_csv('./preprocessing/ICA/H/'+'H_EC_'+str(k)+'_ica.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                    'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1'])
    elif(len(data_frame.columns) == 22):
        data_frame.to_csv('./preprocessing/ICA/H/'+'H_EC_'+str(k)+'_ica.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                    'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1','23A-23R','24A-24R'])
    else:
        print("else error")



    # Plot results
    plt.figure()

    models = [X, S_, H]
    names = ['Observations (mixed signal)',
             'ICA recovered signals',
             'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(3, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.savefig('./preprocessing/H_Plot/H_EC_'+str(k)+'.png')

