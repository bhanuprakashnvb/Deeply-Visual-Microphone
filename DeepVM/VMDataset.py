import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class VideoSoundDataset(Dataset):
    """Dataset containing a Frame with it's corresponding audio signature"""
    def __init__(self, root_dir:str, audio_csv_path:str, transform=None):
        """Initializes a directory with all the videos

        Args:
            root_dir (string): Path to the folder with video files
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.root_dir = root_dir
        self.audioDf = pd.read_csv(audio_csv_path,header=0)
        self.transform = transform
    
    def __len__(self):
        return len(self.audioDf['frame'])

    def get_range(self,idx,ranges:list):
        if len(ranges) == 1:
            return ranges[0]
        else:
            for i in ranges:
                start, fin = i.split('-')
                if idx in range(int(start),int(fin)+1):
                    return f'{start}-{fin}'

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ranges = os.listdir(self.root_dir)
        # based on index, find the correct folder range
        # once the folder range is obtained, create the image path
        # print(self.get_range(idx, ranges), idx)
        root_path = os.path.join(self.root_dir, self.get_range(idx, ranges))
        # root_path = os.path.join(self.root_dir,self.get_range(idx,ranges))
        
        img_name = os.path.join(root_path, f'{self.audioDf.iloc[idx, 0]}.png')
        image = io.imread(img_name)
        image_tensor = torch.from_numpy(image)

        # If it's a color image, permute to convert HxWxC to CxHxW
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        # the index itself can be directly used to fetch the label
        label = self.audioDf.iloc[idx,1]
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        sample = {'image':image_tensor.float(), 'label':label}
        return sample

# test = VideoSoundDataset('./Dataset','../Audio/SampledData.csv',transform=None)
# # for i, item in enumerate(test):
# #     io.imshow(item['image'])
# #     io.show()
# #     print(item['label'])
# print(len(test))