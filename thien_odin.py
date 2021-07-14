import random
import numpy as np
from loader import label_processing
import torch

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
np.random.seed(1542)
### label => one hot encode
one_hot_encoder, classes = label_processing()

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
import random

# from thien_metric import *
plt.ioff()
import pickle
# from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# import albumentations
# from albumentations.pytorch import ToTensorV2
import pickle


def tpr95(iod_scores, ood_scores):
	Y1 = ood_scores
	X1 = iod_scores
	start = np.min(iod_scores)
	end = np.max(iod_scores)
	gap = (end - start) / 10000
	# f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))  #### = true_positive / true_positive + false_positive
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			# print(fpr)
			total += 1
	if total > 0:
		fprNew = fpr / total
		return fprNew
	else:
		return -1




def mse(pred):
	scores = []
	for class_ in classes:
		onehot = one_hot_encoder.transform([class_])[0]
		score = np.square(np.asarray(pred) - np.asarray(onehot))
		scores.append(np.mean(score))
	return np.argmin(scores),np.min(scores) ### index, score

def turning_params(model, iod_loader, ood_loader, temperature=12, noise=0.023):
	##### load model:
	criterion1 = nn.MSELoss()
	stop = 0
	accuracy = 0
	#
	# with torch.no_grad():
	# 	for i, iod_info in enumerate(iod_loader):
	# 		# print(i)
	# 		x = iod_info['img'].to(device)
	# 		target_original = iod_info['label'].to(device)
	# 		target = torch.argmax(target_original, dim=-1)
	# 		target = target.squeeze(-1)
	#
	# 		pred = model(x)
	# 		_, preds = torch.max(pred, 1)  # return maxValue, indice(same with argmax) # y = dimension
	# 		accuracy += torch.sum(preds == target.data)
	# 		print(f'preds {preds}')
	# 		print(f'target {target.data}')
	# print(f'The number of answer : {accuracy} / {len_test}')
	in_scores, out_scores = [], []
	ori_in_scores, ori_out_scores = [], []
	result_path = f'softmax_scores/t-{temperature}_n-{noise}/'
	os.makedirs(result_path, exist_ok=True)
	def detect(info):
		# model.zero_grad()
		img = info['img'].requires_grad_(True).to(device)
		img.retain_grad()
		softmax_out, fc_out = model(img)
		ori_idx, ori_scores = mse(fc_out.data.cpu().numpy())

		pred_labels = torch.zeros_like(fc_out, dtype=torch.float)
		pred_labels[:,ori_idx] = 1.0

		# ###### for original input
		# #### calculate log(softmax) of dense output
		# numpy_dense_outputs = fc_out.data.cpu().numpy()[0]  ## because batch size is 1
		# numpy_dense_outputs = numpy_dense_outputs - np.max(numpy_dense_outputs)
		# # softmax_dense_output_scores = np.exp(numpy_dense_outputs) / np.sum(np.exp(numpy_dense_outputs))
		# mse_score = np.square(numpy_dense_outputs - pred_labels.data.cpu().numpy()[0])
		# ori_scores = np.max(mse_score)
		# with open(os.path.join(result_path,"original_model_in_of_distribution.txt"), 'a+') as f:
		# 	f.write(("{}\n".format(max_score)))

		# with open(os.path.join(result_path,"original_model_out_of_distribution.txt"), 'a+') as f:
		# 	f.write(("{}\n".format(max_score)))

		### calculate scaled output:
		scaled_output = fc_out / temperature
		### calculate loss and update gradient
		loss1 = criterion1(scaled_output, pred_labels)
		loss1.backward()

		### turning gradient
		gradient = torch.ge(img.grad.data, 0)
		### get gradient sign (convert to [-1,1]
		gradient = (gradient.float() - 0.5) * 2
		### normalizing gradient due to normalized input img
		# gradient[0][0] = (gradient[0][0]) / (0.2685)
		# gradient[0][1] = (gradient[0][1]) / (0.2723)
		# gradient[0][2] = (gradient[0][2]) / (0.2711)
		#### for modified input
		new_input = torch.add(img.data, -noise * gradient)
		# model.zero_grad()
		# with torch.no_grad():
		### predict
		new_softmax_output, new_dense_outputs = model(new_input)
		### scale new_output with temperature
		# new_dense_outputs = new_dense_outputs / temperature
		# normalized_outputs = new_dense_outputs.data.cpu().numpy()[0]
		# normalized_outputs = normalized_outputs - np.max(normalized_outputs)  #

		#### create pred label onehot tensor and calculate the mse

		# mse_score = np.square(normalized_outputs - pred_labels.data.cpu().numpy()[0])

		## todo: calculate mse score
		# mse_score = np.square(normalized_outputs - fc_out.data.cpu().numpy()[0])
		# softmax_scores = np.exp(normalized_outputs) / np.sum(np.exp(normalized_outputs))
		# ### take the max softmax score
		# scores = np.max(softmax_scores)
		# scores = np.max(mse_score)
		new_idx, new_scores = mse(new_dense_outputs.data.cpu().numpy())
		return new_scores, ori_scores

	for i, (iod_info, ood_info) in enumerate(zip(iod_loader, ood_loader)):
		if i%200==0:
			print(i)
		stop += 1
		# if stop >=3000:
		# 	break
		# optimizer1.zero_grad()
		scores, ori_scores = detect(iod_info)
		in_scores.append(scores)
		ori_in_scores.append(ori_scores)

		scores_, ori_scores_ = detect(ood_info)

		out_scores.append(scores_)
		ori_out_scores.append(ori_scores_)

	# write_list_to_txt(ori_in_scores, os.path.join(result_path, "original_model_in_of_distribution.txt"))
	# write_list_to_txt(ori_out_scores, os.path.join(result_path, "original_model_out_of_distribution.txt"))
	print('')
	write_list_to_txt(in_scores, os.path.join(result_path, "modified_model_in_of_distribution.txt"))
	write_list_to_txt(out_scores, os.path.join(result_path, "modified_model_out_of_distribution.txt"))

	# false_positive_rate = tpr95(in_scores, out_scores)
	# print(f'false_positive_rate: {false_positive_rate}')
	return in_scores, out_scores


def write_list_to_txt(list, path, mode='w'):
	list = sorted(list)
	with open(path, mode=mode) as file:
		for item in list:
			item = str(float(item))
			file.writelines(item + '\n')


# print(f'wrote to {path} successfully..')

if __name__ == '__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	##### load model:
	model = efficientnet_b1(pretrained=False, num_classes=80).to(device)
	model.to(device)
	checkpoint = torch.load('./checkpoints/' + 'best_model' + '.pth', map_location=device)
	model.load_state_dict(checkpoint['model'])
	model.eval()

	### load tunring data
	#### load label encoder
	one_hot_encoder, classes = label_processing()

	### load data:
	test_dirname = './converse/test'
	unknown_dirname = './converse/unknown'

	### Load train and test set
	list_test, len_test = mklist(root_dir=test_dirname)
	list_unknown, len_unknown = mklist(root_dir=unknown_dirname)
	random.shuffle(list_test)
	random.shuffle(list_unknown)
	### Data loader of train and test set
	iod_dataset = loader.dataset(list_test, one_hot_encoder)
	iod_loader = torch.utils.data.DataLoader(iod_dataset,
	                                         batch_size=1,
	                                         shuffle=False,
	                                         pin_memory=True,
	                                         num_workers=0)

	ood_dataset = loader.dataset(list_unknown, one_hot_encoder)
	ood_loader = torch.utils.data.DataLoader(ood_dataset,
	                                         batch_size=1,
	                                         shuffle=False,
	                                         pin_memory=True,
	                                         num_workers=0)

	temperatures = [1000, 1800]

	for temp in temperatures:
		noise = -0.01
		gap = 0.01/21
		while True:
			noise = noise + gap
			in_scores, out_scores = turning_params(model=model,
			                                       iod_loader=iod_loader,
			                                       ood_loader=ood_loader,
			                                       temperature=temp,
			                                       noise=noise)
			in_scores = sorted(in_scores)
			threshold = in_scores[int(len(in_scores) * 0.95)]
			print(f'temp: {temp} -- noise: {noise} -- threshold: {threshold} -- acc: {sum(score > threshold for score in out_scores)}/{len(out_scores)}')
			if noise >= 0.01:
				break
	### best: temperature 1000, noise: 0.00381