import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os 
import pandas as pd
from PIL import Image
import _testimportmultiple

#---------personlised Dataset------------

class FundusAgeDataset(Dataset):
    def  __init__(self, csv_file, img_dir, transform = None):
        '''Args :
            csv_file (str) : root to the files .csv or .xlsx with culmns 'Filename' and Age'
            img_dir (str): directory with dataset's images.
            transform (callable, optional): transoformation to images
            '''
        if csv_file.endswith('.xlsx'):
            self.data = pd.read_excel
        else:
            self.data = pd.read_csv(csv_file)
        
        self.img_dir = img_dir
        self.transform = transform

        print('dataset initialised')
        print(f'number of images : {len(self.data)}')
        print(f'Exemple : {self.data.head(3)}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ''' return the pair (image, age) for a giving index'''
        row = self.data.iloc[idx]
        filename = row['Filename']
        age = row['Age']

        img_path = os.path.join(self.img_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error with the openning image {filename} : {e}')
            raise

        if self.transform :
            image = self.transform(image)
        
        age_tensor = torch.tensor(age, dtype = torch.float32).unsqueeze(0)

        print(f'[{idx}]{filename}|Age:{age}|Shape image: {image.shape}')

        return image, age_tensor

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
