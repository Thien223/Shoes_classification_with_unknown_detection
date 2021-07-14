from loader import mklist, label_processing
from torch.utils.data import DataLoader
import torch
import pickle
import loader as loader

dirname = '../converse/train/'
gpu_numb=1
batch_size = 256
device = torch.device("cuda:%d"%gpu_numb if torch.cuda.is_available() else "cpu")
#one_hot_encoder = label_processing()
with open('one_hot_encoder.pickle', 'rb') as f:
    one_hot_encoder = pickle.load(f)

list_train, len_train= mklist(root_dir=dirname)

train_dataset = loader.dataset(list_train,one_hot_encoder)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

mean = 0.
std = 0.
mean_rgb = 0
std_rgb = 0
print(len(train_loader.dataset))
with torch.no_grad():    
    for i, sample in enumerate(train_loader):
        
        img = sample['img'].to(device)
        batch_samples_img = img.size(0) # batch size (the last batch can have smaller size!)
        img = img.view(batch_samples_img, img.size(1), -1)
        mean_rgb += img.mean(2).sum(0)
        std_rgb += img.std(2).sum(0)
        
        # images = sample['im_ycbcr'].to(device)
        # batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        # images = images.view(batch_samples, images.size(1), -1)
        # mean += images.mean(2).sum(0)
        # std += images.std(2).sum(0)
        # print('processing:',i,'/',len(train_loader))

    #mean /= len(train_loader.dataset)
    #std /= len(train_loader.dataset)
    mean_rgb /= len(train_loader.dataset)
    std_rgb /= len(train_loader.dataset)
    print("RGB")
    print(mean_rgb)
    print(std_rgb)