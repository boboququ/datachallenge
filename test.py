import numpy as np
import sklearn.metrics
import scipy.io.wavfile
import wave
import os
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from scipy.fftpack import fft

#clustering? 
#separate logistic models for each, take the best probability
#classifiers, random forest, boosting
#Heuristics based on frequency
#Features: FFT 
#Truncate frequency to determine how much is truncated

samples_directory = "./train_data"

classes = ["BassClarinet", "BassTrombone", "BbClarinet", "Cello", "EbClarinet", "Marimba", "TenorTrombone", "Viola", "Violin", "Xylophone"]

num_features = 24

def add_data_sample(file_name, instrument_type):
    file_name = os.path.join(samples_directory, file_name)
    sampling_rate, data = scipy.io.wavfile.read(file_name)

    highest_10_freq = get_highest_10_freq(data, sampling_rate).tolist()
    mfcc = get_mfcc(data, sampling_rate).tolist()
    spectral_centroid = get_spectral_centroid(data, sampling_rate)
    sample = highest_10_freq  + mfcc + [spectral_centroid]
    sample = sample + [instrument_type]
    return sample

def get_highest_10_freq(data,sampling_rate):

    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))
    idx = np.argsort(-np.abs(w))
    freq = freqs[idx[:10]]
    freq_in_hertz = abs(freq * sampling_rate)

    return freq_in_hertz

def get_mfcc(data, sampling_rate):

    mfcc_result = mfcc(data, sampling_rate)
    mfcc_array = np.array(mfcc_result)
    mean = np.mean(mfcc_array, axis = 0)

    return mean

def get_spectral_centroid(data, sampling_rate):
    magnitudes = np.abs(np.fft.rfft(data)) # magnitudes of positive frequencies
    length = len(data)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampling_rate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)

if __name__ == "__main__":
    count = 0
    current_row = 0
    for file_name in os.listdir(samples_directory):
        count = count + 1

    data_frame = np.zeros((count, num_features + 1))
    print("populating_dataframe")
    files_read = 0
    for file_name in os.listdir(samples_directory):
        sample = []
        if classes[0] in file_name:
            sample = add_data_sample(file_name, 0)
        if classes[1] in file_name:
            sample = add_data_sample(file_name, 1)
        if classes[2] in file_name:
            sample = add_data_sample(file_name, 2)
        if classes[3] in file_name:
            sample = add_data_sample(file_name, 3)
        if classes[4] in file_name:
            sample = add_data_sample(file_name, 4)
        if classes[5] in file_name:
            sample = add_data_sample(file_name, 5)
        if classes[6] in file_name:
            sample = add_data_sample(file_name, 6)
        if classes[7] in file_name:
            sample = add_data_sample(file_name, 7)
        if classes[8] in file_name:
            sample = add_data_sample(file_name, 8)
        if classes[9] in file_name:
            sample = add_data_sample(file_name, 9)
        for i in range(len(sample)):
            data_frame[current_row, i] = sample[i]
        current_row = current_row + 1
        files_read = files_read + 1
        print("Files Read: ", files_read)
        print("Read:", file_name)
    print(data_frame)
    np.savetxt("dataset2.csv", data_frame, delimiter=",")
