import math
import cv2 as cv
import numpy as np
import pyrtools as pyr
from scipy import signal
from .spec_sub import normalize_audio

# Function to align two vectors A and B by finding the shift that maximizes their correlation
def alignA_to_B(A: np.array, B: np.array):
    # Compute the cross-correlation between A and the reversed version of B
    # acorb = np.convolve(A, np.flip(B))
    acorb = signal.fftconvolve(A, np.flip(B))

    # Find the index of the maximum value in the cross-correlation, indicating the best alignment
    maxInd = np.argmax(acorb)

    # Calculate the shift amount based on the position of the maximum correlation
    shift = B.size - maxInd
    # Shift vector A by the calculated amount to align it with vector B
    output = np.roll(A, shift)

    return output

# Function to downsample a frame by a given factor
def downsample(frame, downSample_factor):
    # Downsample if the factor is less than 1, otherwise return the original frame
    if downSample_factor < 1:
        scaled_frame = cv.resize(frame, (0,0), fx = downSample_factor, fy = downSample_factor)
    else:
        scaled_frame = frame
    
    return scaled_frame

# Function to convert a frame to grayscale and normalize it
def convert_and_normalize(frame):
    # Convert frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Normalize the grayscale frame to have values between 0.0 and 1.0
    norm_frame = cv.normalize(gray_frame.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return norm_frame

# Function to generate a visual motion-derived sound from a video
def vmSoundFromVideo(videoHandler: cv.VideoCapture, nscale, norientation, downSampleFactor = 1, numFrames = 0, samplingrate = -1):
    # Default sampling rate to video FPS if not specified
    if samplingrate < 0:
        samplingrate = int(videoHandler.get(cv.CAP_PROP_FPS))
    # Default number of frames to total video frames if not specified
    if numFrames == 0:
        numFrames = int(videoHandler.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Read the first frame to initialize processing
    flag, frame = videoHandler.read()
    
    # Process the first frame to use as a reference
    downsampled_frame = downsample(frame, downSampleFactor)
    first_frame = convert_and_normalize(downsampled_frame)

    # Create a reference pyramid from the first frame
    ref_pyramid = pyr.pyramids.SteerablePyramidFreq(first_frame, nscale, norientation-1, is_complex = True)
    ref_pyramid = ref_pyramid.pyr_coeffs

    # Initialize dictionaries to hold signal data for each pyramid band
    signals = {key: list() for key in ref_pyramid.keys()}

    # Process each frame in the video
    while flag:
        # Downsample and process the current frame
        frame = downsample(frame, downSampleFactor)
        processed_frame = convert_and_normalize(frame)

        # Create a pyramid from the processed frame
        pyramid = pyr.pyramids.SteerablePyramidFreq(processed_frame, nscale, norientation-1, is_complex = True)
        pyramid = pyramid.pyr_coeffs

        # Initialize dictionaries for amplitude and phase information
        amplitude = dict()
        phase_info = dict()

        # Calculate amplitude and phase difference for each pyramid band
        for bandIndx, coeffs in pyramid.items():
            amplitude[bandIndx] = np.abs(coeffs)
            ref_coeffs = ref_pyramid[bandIndx]
            phase_info[bandIndx] = np.mod(np.pi + np.angle(coeffs) - np.angle(ref_coeffs), 2 * np.pi) - np.pi

        # Calculate and accumulate the motion signal for each band
        for bandIndx in pyramid.keys():
            ampL = amplitude[bandIndx]
            phaseL = phase_info[bandIndx]

            # Compute the motion signal as the product of phase difference and squared amplitude
            single_motion_signal = np.multiply(phaseL, np.multiply(ampL, ampL))

            # Normalize the motion signal by total amplitude and accumulate
            total_amplitude = np.sum(ampL.flatten())
            signals[bandIndx].append(np.mean(single_motion_signal.flatten()) / total_amplitude)

        # Read the next frame
        flag, frame = videoHandler.read()

    # Initialize an array to accumulate the final audio signal
    audio = np.zeros(numFrames)
    # Align and accumulate signals from each pyramid band
    for signal_ in signals.values():
        aligned_signal = alignA_to_B(np.array(signal_), np.array(signals[(0,0)]))
        # print(aligned_signal)
        audio += aligned_signal
        # print(audio)
    #Applying Filters to the sounds
    bandPass = signal.butter(3, 0.05, btype='highpass', output='sos')
    processed_sound = signal.sosfilt(bandPass, audio)

    #Constrict the sound to a range
    processed_sound = normalize_audio(processed_sound)

    return processed_sound