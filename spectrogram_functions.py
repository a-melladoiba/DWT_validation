import matplotlib
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import os
import warnings
import librosa
from scipy.integrate import cumtrapz
import pandas as pd
from scipy.signal import welch



def compute_psd_welch(data_1, sf, label, folder, muscle, file_name, side, subphase, save_path):

    location = folder.split("/",folder.count('/'))
    subject = location[-2]

    S1, f1 = plt.psd(data_1, NFFT=256, Fs=sf, window=np.hanning(256), noverlap=128)
    plt.close()

    if label == 'plot':
        fig, ax = plt.subplots(figsize=(4,4))

        fig.suptitle(file_name)

        ax.plot(f1, S1)
        ax.set(title= side + ' PSD (' + muscle + ')', xlabel = 'frequency [Hz]', ylabel = 'PSD (dB)')

        ax.set_xlim([0, 200])
        ax.set_xticks(range(20, 201, 20))

        new_save_path = str(save_path + '/' + location[-2])

        plt.savefig(new_save_path + '/PSD_welch_' + subphase + '_' + subject + '_' + muscle + '_' + side + '_' + str(file_name))
        plt.close()
    return S1, f1



def plot_1(data_1, folder, muscle, aff_side, label, file_name, side, save_path):
    location = folder.split("/",folder.count('/'))
    subject = location[-2]

    height = 4
    width = 5

    plt.figure()

    fig, axs = plt.subplots()

    axs.plot(data_1)
    axs.set_title(muscle + ' ' + side)

    # Show the graph
    fig.suptitle(label + ' ' + subject + '. Affected side: ' + aff_side,fontsize=15)

    new_save_path = str(save_path + '/' + location[-2])
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)

    #plt.savefig(new_save_path + '/' + label + '_' + muscle+ '_' + str(file_name))
    plt.savefig(f"{new_save_path}/{label}_{muscle}_{file_name}")
    plt.close()



def spec_librosa(data_1, wind_size, sf, muscle, label, folder, file_name, side, subphase, save_path):
    location = folder.split("/",folder.count('/'))
    subject = location[-2]

    warnings.filterwarnings("ignore")

    stft_audio_1 = librosa.stft(data_1, hop_length = wind_size//4, win_length = wind_size, window = 'hann')

    if label == 'plot':
        fig, axs = plt.subplots()

        D1 = librosa.amplitude_to_db(np.abs(stft_audio_1), ref= np.max)
        img1 = librosa.display.specshow(D1, sr = sf, win_length=wind_size, hop_length = wind_size//4, ax=axs, x_axis = "time", y_axis = "linear", cmap = 'Spectral_r' )

        # Show the graph
        axs.set_title (muscle + ' ' + side + ' spectrogram. Ws = ' + str(wind_size))
        axs.set(xlabel = 'Time (s)', ylabel = 'Frequency (Hz)')

        fig.colorbar(img1, ax=axs, format='%+2.f dB')


        fig.suptitle(subject + '. Spectrogram (ws = ' + str(wind_size) + ')')

        new_save_path = str(save_path + '/' + location[-2])

        plt.savefig(new_save_path + '/Spectrogram_ws_' + str(wind_size) + '_' + subject + '_' + muscle + '_stride_' + side + '_' + subphase + '_' + str(file_name))

    return stft_audio_1



def spec_librosa_3(events, data_1, data_2, mnf_left, mnf_right, mdf_left, mdf_right, win_size, sf, muscle, label, folder, file_name, save_path, value_to_subtract_l, value_to_subtract_r):
    location = folder.split("/",folder.count('/'))
    subject = location[-2]

    stft_audio_1 = librosa.stft(data_1, hop_length = win_size//4, win_length = win_size, window = 'hann')
    y_audio_1 = np.abs(stft_audio_1) ** 2

    stft_audio_2 = librosa.stft(data_2, hop_length = win_size//4, win_length = win_size, window = 'hann')
    y_audio_2 = np.abs(stft_audio_2) ** 2

    if label == 'plot':
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 4), constrained_layout=True)

        D1 = librosa.amplitude_to_db(np.abs(stft_audio_1), ref= np.max)
        img1 = librosa.display.specshow(D1, sr=sf, win_length=win_size, hop_length=win_size // 4, ax=axs[0], x_axis="time", y_axis="linear", cmap='Spectral_r')

        D2 = librosa.amplitude_to_db(np.abs(stft_audio_2), ref= np.max)
        img2 = librosa.display.specshow(D2, sr=sf, win_length=win_size, hop_length=win_size // 4, ax=axs[1], x_axis="time", y_axis="linear", cmap='Spectral_r')

        # Show the graph
        axs[0].set_title (muscle + ' left spectrogram. Ws = ' + str(win_size))
        axs[1].set_title (muscle + ' right spectrogram. Ws = ' + str(win_size))
        axs[0].set(xlabel = 'Time (s)', ylabel = 'Frequency (Hz)')
        axs[1].set(xlabel = 'Time (s)', ylabel = 'Frequency (Hz)')

        fig.colorbar(img1, ax=axs[0], format='%+2.f dB')
        fig.colorbar(img2, ax=axs[1], format='%+2.f dB')

        # Plot the events using matplotlib
        eventos_left_st = []
        eventos_left_sw = []
        eventos_right_st = []
        eventos_right_sw = []
        eventos_left = []
        promedios_left = []
        promedios_right = []
        for curr_col in events.columns:
            if 'heel_strike' in curr_col:
                if 'l_heel_strike' in curr_col:
                    for event in events[curr_col].values:
                        if (event - value_to_subtract_l) >= 0 and (event - value_to_subtract_l) <= axs[0].get_xlim()[1]:
                            axs[0].vlines(event - value_to_subtract_l, ymin=0, ymax=500, color = 'purple', label = 'Heel Strike')
                            eventos_left_st.append(event - value_to_subtract_l)
                        else:
                            eventos_left_st.append(event - value_to_subtract_l)


                elif 'r_heel_strike' in curr_col:
                    for event in events[curr_col].values:
                        if (event - value_to_subtract_r) >= 0 and (event - value_to_subtract_r) <= axs[1].get_xlim()[1]:
                            axs[1].vlines(event - value_to_subtract_r, ymin=0, ymax=500, color = 'purple', label = 'Heel Strike')
                            eventos_right_st.append(event - value_to_subtract_r)
                        else:
                            eventos_right_st.append(event - value_to_subtract_r)

            elif 'toe_off' in curr_col:
                if 'l_toe_off' in curr_col:
                    for event in events[curr_col].values:
                        if (event - value_to_subtract_l) >= 0 and (event - value_to_subtract_l) <= axs[0].get_xlim()[1]:
                            axs[0].vlines(event - value_to_subtract_l, ymin=0, ymax=500, color = 'limegreen', label = 'Toe off')
                            eventos_left_sw.append(event - value_to_subtract_l)
                        else:
                            eventos_left_sw.append(event - value_to_subtract_l)

                elif 'r_toe_off' in curr_col:
                    for event in events[curr_col].values:
                        if (event - value_to_subtract_r) >= 0 and (event - value_to_subtract_r) <= axs[1].get_xlim()[1]:
                            axs[1].vlines(event - value_to_subtract_r, ymin=0, ymax=500, color = 'limegreen', label = 'Toe off')
                            eventos_right_sw.append(event - value_to_subtract_r)
                        else:
                            eventos_right_sw.append(event - value_to_subtract_r) 

        eventos_left = np.concatenate([eventos_left_st,eventos_left_sw])
        eventos_left_sin_nan = sorted(eventos_left[~np.isnan(eventos_left)])
        for i in range(len(eventos_left_sin_nan) - 1):
            promedio_left = (eventos_left_sin_nan[i] + eventos_left_sin_nan[i + 1]) / 2
            promedios_left.append(promedio_left)

        eventos_right = np.concatenate([eventos_right_st,eventos_right_sw])
        eventos_right_sin_nan = sorted(eventos_right[~np.isnan(eventos_right)])

        for i in range(len(eventos_right_sin_nan) - 1):
            promedio_right = (eventos_right_sin_nan[i] + eventos_right_sin_nan[i + 1]) / 2
            promedios_right.append(promedio_right)

        axs[0].plot(promedios_left[0],mnf_left[0],marker='o', label='MNF')
        axs[0].plot(promedios_left[1],mdf_left[1],marker='o',color='black', label='MDF')
        axs[1].plot(promedios_right[0],mnf_right[0],marker='o',label='MNF')
        axs[1].plot(promedios_right[1],mdf_right[1],marker='o',color='black',label='MDF')

        fig.suptitle(file_name[:15] + '. Spectrogram (ws = ' + str(win_size) + ')')
        plt.savefig(save_path + '/' + subject + '/Spectrogram_ws_' + str(win_size) + '_' + file_name[:15] + '_' + muscle)

        plt.close()

    return stft_audio_1, stft_audio_2



def plot_time_series(events,data_1, data_2, sample_rate, value_to_subtract, folder, label, muscle, index):
    location = folder.split("/",folder.count('/'))
    subject = location[-1]

    height = 2
    width = 10
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(width, height), constrained_layout=True)

    time_data = np.arange(0, len(data_1), 1)*(1/sample_rate)

    axs[0].plot(time_data, data_1) #left
    axs[1].plot(time_data, data_2) #right
    # Plot the events using matplotlib
    for curr_col in events.columns:
        if 'Left' in curr_col:
            if 'Foot Strike' in curr_col:
                axs[0].vlines(events[curr_col].values - value_to_subtract, ymin=min(data_1), ymax=max(data_1), color = 'purple')
            elif 'Foot Off' in curr_col:
                axs[0].vlines(events[curr_col].values - value_to_subtract, ymin=min(data_1), ymax=max(data_1), color = 'limegreen')

        elif 'Right' in curr_col:
            if 'Foot Strike' in curr_col:
                axs[1].vlines(events[curr_col].values - value_to_subtract, ymin=min(data_2), ymax=max(data_2), color = 'purple', label = 'Foot strike')
            elif 'Foot Off' in curr_col:
                axs[1].vlines(events[curr_col].values - value_to_subtract, ymin=min(data_2), ymax=max(data_2), color = 'limegreen', label = 'Foot off')
    axs[0].set(xlabel = 'Time (s)', ylabel = 'Amplitude (mV)')
    axs[1].set(xlabel = 'Time (s)', ylabel = 'Amplitude (mV)')

    axs[1].legend()
    fig.suptitle(label + ' ' + subject + '. Affected side: ' + aff_side,fontsize=16)

    plt.savefig(save_path + '/' + subject + '/' + label + '_' + subject + '_' + muscle+ '_' + str(index))

    plt.close()


def compute_mnf(signal, sf):
    freqs, psd = welch(signal, sf, nperseg=1024)

    mnf = np.sum(psd * freqs) / np.sum(psd)
    
    return mnf


def compute_mdf(signal, sf):
    frequencies, psd = welch(signal, sf, nperseg=1024)

    area_freq = cumtrapz(psd, frequencies, initial=0)
    total_power = area_freq[-1]
    median_freq = frequencies[np.where(area_freq >= total_power / 2)[0][0]]

    return median_freq



def compute_mnp(data_1):
    power_spectrogram_1 = librosa.power_to_db(np.abs(data_1)**2)

    mean_power_1 = np.mean(power_spectrogram_1)

    return mean_power_1



def compute_tp(data_1):
    # The PSD represents the power at each frequency bin for each time frame. It's the squared magnitude of the spectrogram values
    power_spectrogram_1 = librosa.power_to_db(np.abs(data_1)**2)

    total_power_1 = np.sum(power_spectrogram_1)

    return total_power_1

