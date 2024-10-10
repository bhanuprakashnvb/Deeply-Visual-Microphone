from argparse import ArgumentParser
import cv2 as cv
import matplotlib.pyplot as plt
from os import path
from scipy.io import wavfile

# Assuming these functions are defined in the mentioned modules
from thonVM.sound_from_video import vmSoundFromVideo
from thonVM.spec_sub import spectral_subtraction


def parse_command_line_arguments():
    # Create a parser for the command line arguments
    parser = ArgumentParser()
    # Define arguments: input video path, output file path, and optional sampling rate
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('-o', '--output', help='Output file path, defaults to recoveredsound.wav', default='recoveredsound.wav')
    parser.add_argument('-s', '--sampling-rate', help='Frame rate of the video, used as sampling rate for sound', type=int, default=None)

    return parser.parse_args()


def display_spectrogram(sound_data, sampling_rate):
    # Plot a spectrogram of the sound data
    plt.figure()
    plt.specgram(sound_data, Fs=sampling_rate, cmap=plt.get_cmap('jet'))
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar().set_label('Power Spectral Density (dB)')
    plt.show()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_command_line_arguments()

    # Open the video file
    video_capture = cv.VideoCapture(args.video_path)
    # Determine the sampling rate from the video's frame rate or the user input
    sampling_rate = round(video_capture.get(cv.CAP_PROP_FPS)) if args.sampling_rate is None else args.sampling_rate

    # Extract sound from the video
    extracted_sound = vmSoundFromVideo(video_capture, 1, 2, downSampleFactor=0.1)

    # Plot and save the original sound
    display_spectrogram(extracted_sound, sampling_rate)
    wavfile.write(args.output, sampling_rate, extracted_sound)

    # Apply spectral subtraction to the extracted sound
    processed_sound = spectral_subtraction(extracted_sound)
    # print(len(processed_sound))
    # print(processed_sound[:100])
    # Plot the spectrogram of the processed sound
    display_spectrogram(processed_sound, sampling_rate)

    # Prepare the filename for the processed sound file
    directory, filename = path.split(args.output)
    base_filename, extension = path.splitext(filename)

    # Save the processed sound
    wavfile.write(path.join(directory, base_filename + '_specsub' + extension), sampling_rate, processed_sound)
