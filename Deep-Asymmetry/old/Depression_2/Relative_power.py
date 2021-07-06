#%%
#https://raphaelvallat.com/bandpower.html
#calculate relative power
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plta
from scipy import signal
from scipy.integrate import simps
import glob
import pandas as pd


# %%

result_list = []
total_list = []

#%%
def bandpower(data, sf, band, window_sec=None,relative=True):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    try:
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp
    except:
        pass
# %%
#alpha + theta -> 4-12
#alpha ->8,13
#theta -> 4,8
######change path
path = './sliced_data/MDD_PCA/*'
file_list_in = glob.glob(path)
list_im = [file for file in file_list_in]
list_im.sort()
print("file_list:{}".format(list_im))

for k in range(len(list_im)):
    print('loading dataset : ',k)
    data_first = pd.read_csv(list_im[k])
    result_list = []
    for n in range(len(data_first.columns)):
        data_cv = data_first.iloc[:, n]
        # print(data_first.columns[n],':',n)
        data = data_cv
        #choosing hz
        result = bandpower(data, 256, [8, 13], window_sec=None)
        print(data_first.columns[n], ':', result)
        result_list.append(result)
    total_list.append(result_list)

print(total_list)

data_frame = pd.DataFrame(total_list)

#%%
#save result as dataframe
#######change name

count = len(data_frame.columns)
data_frame=data_frame.fillna(0)

if (count == 20):
    data_frame.to_csv('./result/MDD_EC_relative_power_pca_alpha_1.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                    'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1'])
elif (count == 22):
    data_frame.to_csv('./result/MDD_EC_relative_power_pca_alpha_1.csv', index=False, header=['Fp1-LE','F3-LE','C3-LE','P3-LE','O1-LE','F7-LE',
                                                                                                    'T3-LE','T5-LE','Fz-LE','Fp2-LE','F4-LE','C4-LE','P4-LE','O2-LE','F8-LE','T4-LE','T6-LE','Cz-LE','Pz-LE','A2-A1','23A-23R','24A-24R'])
else:
    print("else error")


# %%

# %%
