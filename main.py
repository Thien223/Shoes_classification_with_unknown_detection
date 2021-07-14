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
#import albumentations
#from albumentations.pytorch import ToTensorV2
def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0, 1)

def mkdir(dirname1,dirname2,dirname3):
	os.makedirs(dirname1,exist_ok=True)
	os.makedirs(dirname2,exist_ok=True)
	os.makedirs(dirname3,exist_ok=True)

if __name__ == '__main__':

	epoch_max = 500
	gpu_numb = 0
	batch_size = 128
	test_batch_size = 1
	num_classes = 80
	image_save = False
	Pretrained = False
	mkdir('./checkpoints','./pred_res','./test_res')
	train_dirname = './converse/train'
	test_dirname = './converse/test'
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

	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False,pin_memory=True, num_workers=0)

	### Define model and loss
	model = efficientnet_b1(pretrained=False,num_classes=num_classes).to(device)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.005)
	criterion = nn.CrossEntropyLoss()

	print('=== # of train :',len_train)
	print('=== # of test  : ',len_test)

	if Pretrained == True:
		checkpoint = torch.load('./checkpoints/' + 'best_model' + '.pth')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optim'])
		start_epoch = checkpoint['epoch']
		print("=== Model is loaded")
	else:
		start_epoch = 0
	scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

	t=time.perf_counter()
	best_acc = 0
	for epoch in range(start_epoch, epoch_max):
		loss_mean = 0
		for i, sample in enumerate(train_loader):
			x = sample['img'].to(device)
			target = sample['label'].to(device)
			target = torch.argmax(target,dim=-1)
			target = target.squeeze(-1)
			optimizer.zero_grad()
			### forward
			softmax_output, fc_output = model(x)
			# preds = torch.argmax(softmax_output, dim=1).unsqueeze(-1) # return maxValue, indice(same with argmax) # y = dimension
			# print(preds)
			# print(target)
			### backward

			loss = criterion(softmax_output, target)
			loss.backward()
			optimizer.step()
			loss_mean += loss.data
			print(f'Epoch: {epoch}, iter {i}/{len(train_loader)} loss: {loss.data}')

		old_t=t
		t=time.perf_counter()
		mean = loss_mean/len_train
		lr = optimizer.param_groups[0]['lr']
		print(f'Epoch: [{epoch} / {epoch_max}]   Loss: {float(mean):.6f}   LR: {lr:.6f}   Time: {t-old_t:.2f}')

		### Evaluate the model
		accuracy = 0
		if epoch % 10 == 0:
			model.eval()
			f = open('./pred_res/prediction_res_%d.csv'%epoch, 'a', encoding='utf-8')
			wr = csv.writer(f)
			with torch.no_grad():
				for i, sample in enumerate(test_loader):
					x = sample['img'].to(device)
					target_original = sample['label'].to(device)
					target = torch.argmax(target_original,dim=-1)
					target = target.squeeze(-1)
					softmax_output, fc_output = model(x)
					# y = model(x)
					#
					# _, preds = torch.max(y, 1) # return maxValue, indice(same with argmax) # y = dimension
					preds = torch.argmax(softmax_output, dim=-1)
					accuracy += torch.sum(preds == target.data)

					if preds != target:
						### Make a folder to save
						folder_name = sample['folder_name'][0]
						save_path = './test_res/test_%d'%epoch


						### label inverse transform (ex. 69 => MS327SFB)
						preds_label = preds
						one_hot = np.zeros(((1,num_classes)), dtype=int)
						one_hot[0][int(preds)]=1
						np_target = target_original.cpu().detach().numpy()
						preds_label = one_hot_encoder.inverse_transform(one_hot)
						label = one_hot_encoder.inverse_transform(np_target[0])

						### CSV wrong answer

						os.makedirs(save_path,exist_ok=True)
						wrong_answer = open('./test_res/test_%d/wrong_answer.csv'%epoch, 'a', encoding='utf-8')
						wr_wrong_answer = csv.writer(wrong_answer)
						wr_wrong_answer.writerow([preds_label[0],label[0]])
						if image_save==True:
							model_folder = os.path.join(save_path,folder_name)
							os.makedirs(model_folder,exist_ok=True)
							save_image(denorm(x), model_folder+'/output%d_%d_%d.jpg'%(i,preds,target))
					### CSV pred vs target
					wr.writerow([float(preds),float(target)])
			accuracy_percent = accuracy.data/float(len_test)*100.0
			print(f'The number of answer : {accuracy} / {len_test}   Accuracy of test set : ',"%.3f"%accuracy_percent)

			### Save model
			if accuracy_percent > best_acc:
				torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(),'epoch': epoch }, './checkpoints/best_model.pth')
				best_acc = accuracy_percent
			else:
				torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(),'epoch': epoch }, './checkpoints/model_{}_{}.pth'.format(epoch,int(accuracy_percent)))
			f.close()
			wrong_answer.close()
			### CSV accuracy
			f = open('./accuracy.csv', 'a', encoding='utf-8')
			wr = csv.writer(f)
			wr.writerow([float(accuracy_percent)])
			f.close()
			model.train()
		scheduler.step()
