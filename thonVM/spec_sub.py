import numpy as np
from scipy import signal

# This function normalizes and centers an audio signal within the range of -1 to 1
def normalize_audio(audio_signal: np.array):
  # Find the maximum and minimum values of the audio signal
    max_value = np.max(audio_signal)
    min_value = np.min(audio_signal)

  # Normalize only if the audio isn't already in the desired range
    if max_value != 1.0 or min_value != -1.0:
        value_range = max_value - min_value
        # Scale the audio signal to stretch between -1 and 1
        audio_signal = 2 * audio_signal / value_range
        # Recalculate the max value after scaling
        current_max = np.max(audio_signal)
        # Calculate and apply the offset to center the signal
        offset = current_max - 1.0
        audio_signal -= offset

    return audio_signal

# Enhances the audio signal by applying spectral subtraction
# Based on the method proposed by Myers Abraham Davis from MIT
def spectral_subtraction(audio_signal: np.array, quantile=0.5):
  # Perform Short-Time Fourier Transform (STFT) to get the frequency components
    _, _, stft_matrix = signal.stft(audio_signal)

    # Calculate magnitude squared (power) and phase angle of the STFT
    stft_magnitude_squared = np.abs(stft_matrix) ** 2
    stft_phases = np.angle(stft_matrix)

    # Determine the noise floor across the signal by using the quantile value
    noise_floor = np.quantile(stft_magnitude_squared, quantile, axis=-1)

  # Subtract noise floor from the magnitude squared, ensuring non-negative values
    for time_index in range(stft_magnitude_squared.shape[-1]):
        stft_magnitude_squared[:, time_index] -= noise_floor
        stft_magnitude_squared[:, time_index] = np.maximum(stft_magnitude_squared[:, time_index], 0.0)

  # Convert the adjusted magnitude back to its original form
    stft_magnitude = np.sqrt(stft_magnitude_squared)
    new_stft_matrix = np.multiply(stft_magnitude, np.exp(1j * stft_phases))

    # Inverse STFT to convert back to the time domain
    _, enhanced_audio_signal = signal.istft(new_stft_matrix)

    # Normalize the enhanced audio signal to the range [-1, 1]
    enhanced_audio_signal = normalize_audio(enhanced_audio_signal)

    return enhanced_audio_signal