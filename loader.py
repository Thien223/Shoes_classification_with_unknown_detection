import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os, os.path
import glob
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import csv
import random
import cv2
#import albumentations
#from albumentations.pytorch import ToTensorV2

def mklist(root_dir):
    img_list = []
    img_list = sorted(glob.glob(os.path.join(root_dir, '*','*')))
    len_list = len(img_list)
    #random.shuffle(img_list)
    return img_list , len_list

def label_processing():
    classes= []
    with open("./classes.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            classes.append(line[0])
    # print('Classes:\n{}\n'.format(classes))
    # print(f'=== # of class :   {len(classes)}')
    one_hot_encoder = LabelBinarizer()
    one_hot_encoder = one_hot_encoder.fit(classes)
    #labels_onehot_sequences = one_hot_encoder.transform(classes)
    #label = one_hot_encoder.inverse_transform(np.asarray([labels_onehot_sequences[0]]))
    return one_hot_encoder, classes


class dataset(Dataset):
    def __init__(self, img_list, one_hot_encoder):
        self.img_list = img_list
        self.transform= transforms.Compose([
            #transforms.Resize((256,256), interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.6964, 0.6828, 0.6776), (0.2685, 0.2723, 0.2711)) # classes: 100
            ])
        self.one_hot_encoder = one_hot_encoder

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        ### RGB
        img = Image.open(self.img_list[idx]).convert('RGB')

        img = self.transform(img)
        # folder_name=self.img_list[idx].split(os.sep)[3]

        folder_name = os.path.basename(os.path.dirname(self.img_list[idx]))
        # print(folder_name)

        label = self.one_hot_encoder.transform([folder_name])
        aa = {'img':img, 'label':label, 'folder_name':folder_name}
        return aa

