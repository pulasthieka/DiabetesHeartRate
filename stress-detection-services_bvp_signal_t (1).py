import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmin

# load raw ECG signal
bvp_signal = pd.read_csv('BVP.csv');

#frequency
frequency = bvp_signal.iloc[1];

#remove first two rows
bvp_array = bvp_signal.iloc[2:];

#get row data of a minute
bvp_numpy = bvp_array[270:2190].to_numpy()

#numpy array details
#print("--------- Numpy array ----------");
#print(bvp_numpy.ndim);
#print(format(bvp_numpy.shape) + "\n");

#convert to 1d array
bvp_numpy_one_d_array = bvp_numpy[:,0]

#get peaks of the signal
peaks, _ = find_peaks(bvp_numpy_one_d_array, height = 0, distance = 30)

#print("----peaks------")
#print(peaks)

plt.plot(bvp_numpy_one_d_array)
plt.plot(peaks, bvp_numpy_one_d_array[peaks], "*", color='r', )
#plt.show()

#distance of peaks
peak_distance = np.diff(peaks)

#average distance of peaks in a minute
average_peak_distance = sum(peak_distance)/len(peak_distance)
print("Average peak distance - " + format(average_peak_distance) + "\n");

#creat array of hr and hrv
heart_rate_array = [];
heart_rate_variability_array = [];
nn_50 = 0
#
print("peak")
print(len(peak_distance))
print(peak_distance)
#

for x in range(1, len(peak_distance)):
    #heart rate
    heart_rate = 60 * frequency / peak_distance[x-1];
    heart_rate_array.append(heart_rate);

    #heart rate variability
    heart_rate_variability = abs((peak_distance[x] - peak_distance[x-1]) * 1000 / 64);
    heart_rate_variability_array.append(int(heart_rate_variability));

    #no. of hrv intervals differ more than 50ms
    if (heart_rate_variability > 50):
        nn_50 += 1;

#mean and std - heart rate - BPM
heart_rate_mean = np.mean(heart_rate_array);
heart_rate_std = np.std(heart_rate_array);

print("Heart rate mean (BPM) - " + format(heart_rate_mean));
print("Heart rate std (BPM)  - " + format(heart_rate_std) + "\n");

#mean and std - heart rate variability per minute
heart_rate_variability_mean = np.mean(heart_rate_variability_array)
heart_rate_variability_std = np.std(heart_rate_variability_array)

print("Heart rate variability mean (ms) - " + format(heart_rate_variability_mean));
print("Heart rate variability std (ms)  - " + format(heart_rate_variability_std) + "\n");

#no. and percentage of hrv intervals differ more than 50ms
pNN_50 = nn_50 / len(peak_distance);

print("NN 50  - " + format(nn_50));
print("pNN 50 - " + format(pNN_50) + "\n");

#Triangular interpolation index
peaks_min = argrelmin(bvp_numpy_one_d_array)

min_row = [];
dif_array = [];
dif = 36;

for x in range(1, len(peaks_min[0])):
    if (bvp_numpy_one_d_array[peaks_min[0][x-1]] < 0):
        if (x-1 != 0):
            dif = abs(peaks_min[0][x-1] - peaks_min[0][x-2]);
        if (dif > 12):
            min_row.append(peaks_min[0][x-1]);
            dif = dif * 1000 / 64;
            dif_array.append(dif);

plt.plot(bvp_numpy_one_d_array)
plt.plot(min_row, bvp_numpy_one_d_array[min_row], "*", color='r', )
#plt.show()

TINN = np.mean(dif_array);
print("TINN (ms) - " + format(TINN) + "\n");

sum_hrv_square= 0;

#rmsHRV
for x in range(1, len(heart_rate_variability_array)):
    sum_hrv_square = (heart_rate_variability_array[x-1] ** 2) + sum_hrv_square;

average_hrv_square = sum_hrv_square / len(heart_rate_variability_array);
rmsHRV = np.sqrt(average_hrv_square);
print("rmsHRV (ms) - " + format(rmsHRV) + "\n");

##
from scipy.interpolate import interp1d

rr_y = heart_rate_variability_array;
RR_x = peaks[2:];

RR_x_new = np.linspace(RR_x[0],RR_x[-1],RR_x[-1])
f = interp1d(RR_x, rr_y, kind='cubic')

print(len(rr_y))
print(RR_x)
plt.plot(RR_x, rr_y)
plt.plot(RR_x_new, f(RR_x_new), label="Interpolated", color='red')
#plt.plot(heart_rate_variability_array)
plt.legend();
plt.show()

##
signal_len = len(bvp_numpy_one_d_array);
frq = np.fft.fftfreq(signal_len, d=(1/64))
frq = frq[range(int(signal_len/2))]


Y = np.fft.fft(f(RR_x_new))/signal_len
Y = Y[range(int(signal_len/2))]

plt.plot(frq, abs(Y));
plt.show()
'''
frq = frq[range(signal_len//2)] #Get single side of the frequency range

#Do FFT
Y = np.fft.fft(f(RR_x_new))/signal_len #Calculate FFT
Y = Y[range(signal_len/2)] #Return one side of the FFT

#Plot
plt.title("Frequency Spectrum of Heart Rate Variability")
plt.xlim(0,0.6) #Limit X axis to frequencies of interest (0-0.6Hz for visibility, we are interested in 0.04-0.5)
plt.ylim(0, 50) #Limit Y axis for visibility
plt.plot(frq, abs(Y)) #Plot it
plt.xlabel("Frequencies in Hz")
plt.show()
'''

#trapezoidal intergration or trapz to find the area of freq
#ultra low frequency
ulf = np.trapz(abs(Y[(frq>=0.01) & (frq<=0.04)]))
print("Ultra low frequency  : ", ulf)

#low frequency
lf = np.trapz(abs(Y[(frq>=0.04) & (frq<=0.15)]))
print("Low frequency        : ", lf)

#high frequency
hf = np.trapz(abs(Y[(frq>=0.15) & (frq<=0.4)]))
print("High frequency       : ", hf)

#Ultra high frequncy
uhf = np.trapz(abs(Y[(frq>=0.4) & (frq<=1)]))
print("Ultra high frequency : ", uhf)

#LF and HF ratio
lf_hf_ratio = lf / hf;
print("LF and HF ratio      : ", lf_hf_ratio);

#sum of frequency components - uhf, hf, lf and ulf
sum_frq_components = ulf + lf + hf + uhf;
print("sum of frequency components : ", sum_frq_components);

#relative power of frequncy components
#ultra low frequency
rel_ulf = ulf / sum_frq_components;
print("Relative power of ULF : ", rel_ulf)

#low frequency
rel_lf = lf / sum_frq_components;
print("Relative power of LF  : ", rel_lf)

#high frequency
rel_hf = hf / sum_frq_components;
print("Relative power of HF  : ", rel_hf)

#ultra hifh frequency
rel_uhf = uhf / sum_frq_components;
print("Relative power of UHF : ", rel_uhf)
##

#normalisrd hf and lf components
norm_hf = hf / lf + hf
norm_lf = lf / lf + hf

print("Normalised HF : ", norm_hf);
print("Normalised LF : ", norm_lf);
