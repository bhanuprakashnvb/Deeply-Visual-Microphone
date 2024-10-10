import pandas as pd
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import os

def subsample_audio(audio_path, num_samples):
    """ Subsample audio data to match the number of required samples. """
    sample_rate, data = wavfile.read(audio_path)
    
    if len(data) < num_samples:
        # If number of samples in the audio is lesser than the number of samples in the video
        indices = np.linspace(0, len(data) - 1, num=num_samples, endpoint=True).astype(int)

    else:
        # If number of samples in the audio is larger than the number of samples in the video
        interpolation_function = interp1d(np.linspace(0, len(data) - 1, num=num_samples), 
                                        np.arange(num_samples), 
                                        kind='linear', 
                                        fill_value="extrapolate")
        indices = interpolation_function(np.arange(num_samples)).astype(int)
    
    # Subsample based on situation for each video and audio pair
    subsampled_data = data[indices]

    mean = np.mean(subsampled_data)
    std = np.std(subsampled_data)
    subsampled_data = (subsampled_data - mean) / std


    min_val = np.min(subsampled_data)
    max_val = np.max(subsampled_data)
    subsampled_data = (subsampled_data - min_val) / (max_val - min_val)
    print(subsampled_data)
    return subsampled_data



def process_videos(metadata_file, audio_folder, output_csv):
    # Load metadata
    df = pd.read_csv(metadata_file)
    
    # Prepare the DataFrame to collect audio samples data
    audio_samples_df = pd.DataFrame()
    frames_loaded = 0
    
    for index, row in df.iterrows():
        video_name = row['video_name']
        frame_rate = row['frame_rate']
        num_frames = row['num_frames']
        
        audio_path = os.path.join(audio_folder, f"{video_name}.wav")
        if not os.path.exists(audio_path):
            print(f"Audio file for {video_name} not found.")
            continue
        
        # Subsample audio
        subsampled_data = subsample_audio(audio_path, num_frames)
        # subsampled_data = subsampled_data/32767

        # Save subsampled audio to a .wav file
        wavfile.write(f'../Audio/{video_name}_subsampled_0To1.wav', frame_rate, subsampled_data)

        temp_df = pd.DataFrame({
            'frame': [f'frame_{i+frames_loaded}' for i in range(num_frames)],
            'value': subsampled_data
        })

        audio_samples_df = pd.concat([audio_samples_df, temp_df], ignore_index=True)
        frames_loaded += num_frames
    
    # Save to CSV
    audio_samples_df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Usage example
metadata_file = '../Audio/metadata.csv'  # This file should have columns: video_name, frame_rate, num_frames
audio_folder = '../Audio'
output_csv = '../Audio/SampledData.csv'
process_videos(metadata_file, audio_folder, output_csv)



