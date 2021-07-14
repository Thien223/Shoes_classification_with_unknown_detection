from torch.utils.data import DataLoader
import loader as loader
import torch
#from efficientnet_pytorch import EfficientNet
#from efficientnet_pytorch.model import EfficientNet
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from loader import mklist, label_processing
from effi import efficientnet_b1, preprocess
import time
from torchvision.utils import save_image
import os
import numpy as np
import csv
import matplotlib.pylab as plt
plt.ioff()
import pickle
#from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,  precision_score, recall_score, f1_score, accuracy_score
#import albumentations
#from albumentations.pytorch import ToTensorV2
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def random_list(i):
    import random
    list = []
    rand_numb = random.randint(1,7)
    for _ in range(7):
        while rand_numb in list:
            rand_numb = random.randint(1,7)
        list.append(rand_numb)
    for i_ in range(7-i):
        list.pop()
    list[0]=0
    return list

def mkdir(dirname1,dirname2,dirname3):
    os.makedirs(dirname1,exist_ok=True)
    os.makedirs(dirname2,exist_ok=True)
    os.makedirs(dirname3,exist_ok=True)

if __name__ == '__main__':
    
    epoch_max = 500
    gpu_numb = 1
    batch_size = 50
    test_batch_size = 1
    num_classes = 80
    image_save = True
    Pretrained = True
    mkdir('./checkpoints','./pred_res','./test_res')
    train_dirname = 'converse/train'
    test_dirname = 'converse/test'
    device = torch.device("cuda:%d"%gpu_numb if torch.cuda.is_available() else "cpu")

    ### Load train and test set
    list_train, len_train= mklist(root_dir=train_dirname)
    list_test, len_test = mklist(root_dir=test_dirname)

    ### label => one hot encode
    one_hot_encoder = label_processing()
    ### Save
    with open('./one_hot_encoder.pickle', 'wb') as f:
        pickle.dump(one_hot_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
    # ### Load        
    # with open('one_hot_encoder.pickle', 'rb') as f:
    #     one_hot_encoder = pickle.load(f)
   
    ### Data loader of train and test set
    train_dataset = loader.dataset(list_train,one_hot_encoder)
    test_dataset = loader.dataset(list_test,one_hot_encoder)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                    shuffle=True,pin_memory=device, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_batch_size,
                    shuffle=False,pin_memory=device, num_workers=4)
    ### Define model and loss    
    model = efficientnet_b1(pretrained=False,num_classes=num_classes).to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    print('=== # of train :',len_train)
    print('=== # of test  : ',len_test)

    if Pretrained == True:
        checkpoint = torch.load('./checkpoints/' + 'best_model' + '.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print("=== Model is loaded")

    best_acc = 0
    with torch.no_grad():

        ### Evaluate the model 
        accuracy = 0    
        model.eval()
        p = []
        t = []
        #f = open('./pred_res/prediction_res_%d.csv'%epoch, 'a', encoding='utf-8')
        #wr = csv.writer(f)
        with torch.no_grad():    
            for i, sample in enumerate(test_loader):
                print(i)
                x = sample['img'].to(device)
                #x_LBP = sample['img_LBP'].to(device)
                #x_YCBCR = sample['img_YCBCR'].to(device)
                #x_cat = torch.cat((x,x_YCBCR,x_LBP),dim=1)
                #x = torch.cat((x_original,x_YCbCr),dim=1)
                target_original = sample['label'].to(device)
                target = torch.argmax(target_original,dim=-1)
                target = target.squeeze(-1)

                y = model(x)

                _, preds = torch.max(y, 1) # return maxValue, indice(same with argmax) # y = dimension
                accuracy += torch.sum(preds == target.data)
                p.append(int(preds))
                t.append(int(target))
                


                ### pred와 GT가 다를 때 실제 label(model명)을 csv에 기록하고 해당 image를 저장하기 위한 코드
                # if preds != target:
                #     ### Make a folder to save
                #     folder_name = sample['folder_name'][0]
                #     save_path = './test_res/test_%d'%epoch
                #     model_folder = os.path.join(save_path,folder_name)
                #     os.makedirs(model_folder,exist_ok=True)

                #     ### label inverse transform (ex. 69 => MS327SFB)
                #     preds_label = preds
                #     one_hot = np.zeros(((1,num_classes)), dtype=int)
                #     one_hot[0][int(preds)]=1
                #     np_target = target_original.cpu().detach().numpy()
                #     preds_label = one_hot_encoder.inverse_transform(one_hot)
                #     label = one_hot_encoder.inverse_transform(np_target[0])

                #     ### CSV wrong answer
                #     wrong_answer = open('./test_res/test_%d/wrong_answer.csv'%epoch, 'a', encoding='utf-8')
                #     wr_wrong_answer = csv.writer(wrong_answer)
                #     wr_wrong_answer.writerow([preds_label[0],label[0]])
                #     if image_save==True:
                #         save_image(denorm(x), model_folder+'/output%d_%d_%d.jpg'%(i,preds,target))
                # ### CSV pred vs target
                # wr.writerow([float(preds),float(target)])
        accuracy_percent = accuracy.data/float(len_test)*100.0
        print(f'The number of answer : {accuracy} / {len_test}   Accuracy of test set : ',"%.3f"%accuracy_percent)

        ### confusion matrix
        preds = p
        target = t
        cf = confusion_matrix(preds, target)
        print(cf)
        
        f = open('./confusion_matrix.csv', 'a', encoding='utf-8')
        
        wr = csv.writer(f)
        
        wr.writerows(cf) 

        p = precision_score(preds, target,average='macro')
        print('p',p)
        r = recall_score(preds, target,average='macro')
        print('r',r)
        f1 = f1_score(preds, target,average='macro')
        print('f1',f1)
        acc= accuracy_score(preds,target)
        print('acc',acc)

