import numpy as np
import numpy.fft as FT
import matplotlib.pyplot as plt
import soundfile as sf

'''
All of the lines surrounded by arrows are for generating plots, they can be simply ignored.
'''


x_t, f_s = sf.read('Audio/Moaaz.wav')

x_t = x_t.mean(axis=1) if x_t.ndim > 1 else x_t # Convert to mono
t = np.linspace(0, len(x_t) / f_s, num=len(x_t)) # Define time domain

print(f"The signal's energy in the time domain is: {np.sum(x_t**2)}")

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
time_fig, time_ax = plt.subplots()
time_ax.plot(t, x_t, color='k')
time_ax.set_xlabel('Time [s]')
time_ax.set_ylabel('Amplitude')
time_ax.set_title('Audio Signal')
time_fig.savefig('Plots/Moaaz_Signal.png')
plt.close(time_fig)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


X_f = FT.fftshift(FT.fft(x_t)) # FT centered around zero frequency
f = np.linspace(-f_s/2, f_s/2, len(X_f), endpoint=False) # Define frequency domain

print(f"The signal's energy in the frequency domain is: {np.sum(np.abs(X_f)**2)}")
print(f"The signal's avg power in the frequency domain is: {np.mean(np.abs(X_f)**2)}")

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
freq_fig, freq_ax = plt.subplots(1,2)
freq_fig.set_size_inches(12.8, 4.8)
freq_ax[0].plot(f, np.abs(X_f), color='b')
freq_ax[0].set_xlabel('Frequency [Hz]')
freq_ax[0].set_ylabel('Magnitude')
freq_ax[0].set_title('Frequency Spectrum')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

r = X_f.size // 50

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# Plotting a zoomed-in version of the frequency spectrum around zero frequency
freq_ax[1].plot(f[len(X_f)//2 - r:len(X_f)//2 + r], np.abs(X_f[len(X_f)//2 - r:len(X_f)//2 + r]), color='b')
freq_ax[1].set_xlabel('Frequency [Hz]')
freq_ax[1].set_ylabel('Magnitude')
freq_ax[1].set_title('Frequency Spectrum')
freq_fig.savefig('Plots/Moaaz_Frequency_Spectrum.png')
plt.close(freq_fig)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# --- Ideal Notch Filter ---

index_max = np.argmax(np.abs(X_f)) # the index of the maximum magnitude in the frequency spectrum
f_max = f[index_max]

print(f"The frequency with the maximum magnitude is: {f[index_max]} Hz")


df = f[1] - f[0] # The smallest frequency increment

# ===== 50 Hz Notch Filter =====

bw = 50 # Notch filter bandwidth
window = bw//df
lower_bound = int(index_max - window//2)
upper_bound = int(index_max + window//2)

# Removing the desired frequency band
Notch_50Hz = X_f.copy()
Notch_50Hz[lower_bound:upper_bound] = 0
Notch_50Hz[len(X_f) - upper_bound:len(X_f) - lower_bound] = 0


filtered_Xf_fig, filtered_Xf_ax = plt.subplots(1, 3)
filtered_Xf_fig.set_size_inches(19.2, 4.8)

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_Xf_ax[0].plot(f[len(X_f)//2 - r:len(X_f)//2 + r], np.abs(Notch_50Hz[len(X_f)//2 - r:len(X_f)//2 + r]), color='b')
filtered_Xf_ax[0].set_xlabel('Frequency [Hz]')
filtered_Xf_ax[0].set_ylabel('Magnitude')
filtered_Xf_ax[0].set_title('50Hz Bandwidth Filtered Frequency Spectrum')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Applying IFFT and saving as .wav file
x_t_50Hz_filtered = FT.ifft(FT.ifftshift(Notch_50Hz)).real
sf.write('Audio/Moaaz_50Hz_Filtered.wav', x_t_50Hz_filtered, f_s)

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_xt_fig ,filtered_xt_ax = plt.subplots(1,3)
filtered_xt_fig.set_size_inches(19.2, 4.8)

filtered_xt_ax[0].plot(t, x_t_50Hz_filtered, color='k')
filtered_xt_ax[0].set_xlabel('Time [s]')
filtered_xt_ax[0].set_ylabel('Amplitude')
filtered_xt_ax[0].set_title('50Hz Bandwidth Filtered Audio')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ===== 100 Hz Notch Filter =====

bw = 100 # Notch filter bandwidth
window = bw//df
lower_bound = int(index_max - window//2)
upper_bound = int(index_max + window//2)

# Removing the desired frequency band
Notch_100Hz = X_f.copy()
Notch_100Hz[lower_bound:upper_bound] = 0
Notch_100Hz[len(X_f) - upper_bound:len(X_f) - lower_bound] = 0

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_Xf_ax[1].plot(f[len(X_f)//2 - r:len(X_f)//2 + r], np.abs(Notch_100Hz[len(X_f)//2 - r:len(X_f)//2 + r]), color='b')
filtered_Xf_ax[1].set_xlabel('Frequency [Hz]')
filtered_Xf_ax[1].set_ylabel('Magnitude')
filtered_Xf_ax[1].set_title('100Hz Bandwidth Filtered Frequency Spectrum')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Applying IFFT and saving as .wav file
x_t_100Hz_filtered = FT.ifft(FT.ifftshift(Notch_100Hz)).real
sf.write('Audio/Moaaz_100Hz_Filtered.wav', x_t_100Hz_filtered, f_s)

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_xt_ax[1].plot(t, x_t_100Hz_filtered, color='k')
filtered_xt_ax[1].set_xlabel('Time [s]')
filtered_xt_ax[1].set_ylabel('Amplitude')
filtered_xt_ax[1].set_title('100Hz Bandwidth Filtered Audio')
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ===== 200 Hz Notch Filter =====

bw = 200 # Notch filter bandwidth
window = bw//df
lower_bound = int(index_max - window//2)
upper_bound = int(index_max + window//2)

# Removing the desired frequency band
Notch_200Hz = X_f.copy()
Notch_200Hz[lower_bound:upper_bound] = 0
Notch_200Hz[len(X_f) - upper_bound:len(X_f) - lower_bound] = 0

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_Xf_ax[2].plot(f[len(X_f)//2 - r:len(X_f)//2 + r], np.abs(Notch_200Hz[len(X_f)//2 - r:len(X_f)//2 + r]), color='b')
filtered_Xf_ax[2].set_xlabel('Frequency [Hz]')
filtered_Xf_ax[2].set_ylabel('Magnitude')
filtered_Xf_ax[2].set_title('200Hz Bandwidth Filtered Frequency Spectrum')
filtered_Xf_fig.savefig('Plots/Moaaz_Frequency_Spectrum_Filtered.png')
plt.close(filtered_Xf_fig)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Applying IFFT and saving as .wav file
x_t_200Hz_filtered = FT.ifft(FT.ifftshift(Notch_200Hz)).real
sf.write('Audio/Moaaz_200Hz_Filtered.wav', x_t_200Hz_filtered, f_s)

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
filtered_xt_ax[2].plot(t, x_t_200Hz_filtered, color='k')
filtered_xt_ax[2].set_xlabel('Time [s]')
filtered_xt_ax[2].set_ylabel('Amplitude')
filtered_xt_ax[2].set_title('200Hz Bandwidth Filtered Audio')
filtered_xt_fig.savefig('Plots/Moaaz_Time_Signal_Filtered.png')
plt.close(filtered_xt_fig)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^