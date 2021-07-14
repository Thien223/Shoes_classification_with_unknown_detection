from torch.utils.data import DataLoader
import loader as loader
import torch
# from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.model import EfficientNet
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
from random import randint
plt.ioff()
import pickle


# import albumentations
# from albumentations.pytorch import ToTensorV2
def denorm(x):
	out = (x + 1) / 2
	return out.clamp(0, 1)


def mkdir(dirname1, dirname2, dirname3):
	os.makedirs(dirname1, exist_ok=True)
	os.makedirs(dirname2, exist_ok=True)
	os.makedirs(dirname3, exist_ok=True)


def val(model, valoader, best_acc=0, write=False):
	### Evaluate the model

	model.eval()
	accuracy = 0
	### Make a folder to save
	save_path = './test_res/test_%d' % epoch
	os.makedirs(save_path, exist_ok=True)
	wrong_answer = []
	all_answer = []
	val_losses = 0
	from  torch.nn.functional import mse_loss
	for i, sample in enumerate(valoader):
		x = sample['img'].to(device)
		target_original = sample['label'].to(device)
		target_onehot = torch.argmax(target_original, dim=-1)
		target_onehot = target_onehot.squeeze(-1)
		with torch.no_grad():
			softmax_output, fc_output = model(x)
			preds = torch.argmax(fc_output, dim=-1)
			accuracy += torch.sum(preds == target_onehot.data)
			val_losses += mse_loss(fc_output, target_original.squeeze(1))
		if write and (preds != target_onehot):
			### label inverse transform (ex. 69 => MS327SFB)
			one_hot = np.zeros_like(softmax_output, dtype=int)
			one_hot[0][int(preds)] = 1
			np_target = target_original.cpu().detach().numpy()
			preds_label = one_hot_encoder.inverse_transform(one_hot)
			label = one_hot_encoder.inverse_transform(np_target[0])
			wrong_answer.append([preds_label[0], label[0]])

		### CSV pred vs target
		all_answer.append([float(preds), float(target_onehot)])

	accuracy_percent = 100*accuracy / float(len_test)
	print('')
	print(f'validation loss mean: {val_losses/float(len_test)}')
	print(f'The number of answer : {accuracy} / {len_test}   Accuracy of test set : ', "%.3f" % accuracy_percent)
	print('')

	### Save model
	if accuracy_percent > best_acc:
		torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch}, './checkpoints/best_model.pth')
		best_acc = accuracy_percent
	else:
		torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch}, './checkpoints/model_{}_{}.pth'.format(epoch, int(accuracy_percent)))

	if write:
		### CSV wrong answer
		with open(f'./test_res/test_{epoch}/wrong_answer.csv', 'w', encoding='utf-8') as wr_file:
			wr_writer = csv.writer(wr_file)
			for row in wrong_answer:
				wr_writer.writerow(row)
		### CSV all answer
		with open(f'./pred_res/prediction_res_{epoch}.csv', 'w', encoding='utf-8') as all_file:
			all_writer = csv.writer(all_file)
			for row in all_answer:
				all_writer.writerow(row)
		### CSV accuracy
		with open('./accuracy.csv', 'a+', encoding='utf-8') as f:
			wr = csv.writer(f)
			wr.writerow([float(accuracy_percent)])
	return accuracy_percent, best_acc




if __name__ == '__main__':

	epoch_max = 1200
	gpu_numb = 0
	batch_size = 36 if torch.cuda.is_available() else 2
	test_batch_size = 1
	num_classes = 80
	image_save = False
	Pretrained = False
	mkdir('./checkpoints', './pred_res', './test_res')
	train_dirname = './converse/train'
	test_dirname = './converse/test'
	device = torch.device("cuda:%d" % gpu_numb if torch.cuda.is_available() else "cpu")

	### Load train and test set
	list_train, len_train = mklist(root_dir=train_dirname)
	list_test, len_test = mklist(root_dir=test_dirname)

	### label => one hot encode
	one_hot_encoder, classes = label_processing()
	### Save
	with open('./one_hot_encoder.pickle', 'wb') as f:
		pickle.dump(one_hot_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
	# ### Load
	# with open('one_hot_encoder.pickle', 'rb') as f:
	#     one_hot_encoder = pickle.load(f)

	### Data loader of train and test set
	train_dataset = loader.dataset(list_train, one_hot_encoder)
	test_dataset = loader.dataset(list_test, one_hot_encoder)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=0)

	### Define model and loss
	model = efficientnet_b1(pretrained=False, num_classes=num_classes).to(device)
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.005)
	criterion1 = nn.CrossEntropyLoss()
	criterion2 = nn.MSELoss()

	print('=== # of train :', len_train)
	print('=== # of test  : ', len_test)

	if Pretrained:
		checkpoint = torch.load('./checkpoints/' + 'best_model' + '.pth')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optim'])
		start_epoch = checkpoint['epoch']
		print("=== Model is loaded")
	else:
		start_epoch = 0
	scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

	t = time.perf_counter()
	best_acc = 0
	for epoch in range(start_epoch, epoch_max):
		loss_mean = 0
		model.train()
		for i, sample in enumerate(train_loader):
			x = sample['img'].to(device)
			target_ori = sample['label'].to(device)
			target = torch.argmax(target_ori, dim=-1)
			target = target.squeeze(-1)
			optimizer.zero_grad()

			softmax_output, fc_output = model(x)

			### cross entropy loss
			# loss = criterion1(softmax_output, target)
			# loss.backward()

			#### mse loss:
			loss = criterion2(fc_output, target_ori.squeeze().float())
			loss.backward()
			loss_mean += loss.data

			optimizer.step()

			rand = randint(0,100)
			print(f'loss: {loss}')
			if rand%24==0:
				print('')
				print(f'model output: {fc_output}')
				print(f'target: {target_ori}')
				print('')


		old_t = t
		t = time.perf_counter()
		mean = loss_mean / len_train
		lr = optimizer.param_groups[0]['lr']
		print(f'Epoch: [{epoch} / {epoch_max}]   Loss: {float(mean):.6f}   LR: {lr:.6f}   Time: {t - old_t:.2f}')

		acc, best_acc = val(model, valoader=test_loader, best_acc=best_acc)
		scheduler.step()
