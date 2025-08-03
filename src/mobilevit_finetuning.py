import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os 
import pandas as pd
from PIL import Image
import _testimportmultiple
import timm

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

        #print(f'[{idx}]{filename}|Age:{age}|Shape image: {image.shape}')

        return image, age_tensor

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "data", "metadata_healthy_only.xlsx")
    image_dir = os.path.join(current_dir,'data','images')

    df = pd.read_excel(excel_path)

    
    dataset = FundusAgeDataset(df, image_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader

# ----------- Charger le modèle MobileViT -----------
def get_model():
    model = timm.create_model("mobilevit_s", pretrained=True, num_classes=1)
    return model

# ----------- Boucle d'entraînement -----------
def train(model, dataloader, device, epochs=20, lr=1e-4):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, ages in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, ages = images.to(device), ages.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")

    return model

# ----------- Script -----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_data()
    model = get_model()

    model = train(model, dataloader, device, epochs=30, lr=1e-4)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mobilevit_finetuned.pth")
    print(" Modèle sauvegardé dans models/mobilevit_finetuned.pth")

if __name__ == "__main__":
    main()
