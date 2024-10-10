import cv2
import os

def extract_frames(video_folder, output_folder):
    start_frame = 0  #Initial frame index

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        
        #Load the video
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Calculate the range for this video
        end_frame = start_frame + num_frames - 1
        range_folder = f"{start_frame}-{end_frame}"
        range_folder_path = os.path.join(output_folder, range_folder)
        
        #Create a folder for the current range if it doesn't exist
        if not os.path.exists(range_folder_path):
            os.makedirs(range_folder_path)
        
        frame_idx = start_frame
        reference_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (224, 224))

            if frame_idx == start_frame:
                #Store the first frame as the reference frame, also resized
                reference_frame = frame_resized

            #Concatenate the reference frame with the current frame
            concatenated_frame = cv2.hconcat([reference_frame, frame_resized])

            #Save each frame
            frame_file = os.path.join(range_folder_path, f"frame_{frame_idx}.png")
            cv2.imwrite(frame_file, concatenated_frame)
            frame_idx += 1
        
        #Update the start_frame for the next video
        start_frame = end_frame + 1
        
        #Release the video capture object
        cap.release()

#Usage
video_folder = '../Videos'
output_folder = './Dataset'
extract_frames(video_folder, output_folder)