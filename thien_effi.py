import argparse
import random

import numpy as np
import torch

seed = 1454
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device_ids = [0, 1]
from effi import efficientnet_b3
from torch.utils.data import DataLoader
import loader as loader
import torch
import torch.optim as optim
import torch.nn as nn
from loader import mklist, label_processing
import os
import matplotlib.pylab as plt
from random import randint

plt.ioff()
import pickle


######################
####### modules#######
######################
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Conv2d(torch.nn.Module):
    '''
    Usual 2D convolutional neural network. Included the batch normalization and activation function.
    If the batch normalization is not necessary, use fuseforward instead of forward function.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, activation=None, w_init_gain='linear'):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        activation = activation.strip().replace(' ', '').lower() if activation is not None else activation
        assert activation in ['relu', 'silu', 'leakyrelu', 'tank', 'sigmoid', 'relu6', None], "activation function must be one of ['relu','relu6','silu','leakyrelu','tank','sigmoid']"
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'relu6':
            self.activation = torch.nn.ReLU6()
        else:
            self.activation = None
        ### initialized model weights
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        '''
        forward the process.
        Parameters
        ----------
        x input of model

        Returns
        -------
        output of model
        '''
        if self.activation is not None:
            return self.activation(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))

    def fuseforward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        else:
            return self.conv(x)


class Inception_Block(torch.nn.Module):
    def __init__(self, in_channels=8, bottneck_out_channel=8, conv_out_channels=32, kernels=None, stride=None, reduce=None):
        """
        reduce: whether reduce dimension of input
        """
        super(Inception_Block, self).__init__()
        if kernels is None:
            kernels = [11, 21, 41]
        if stride is None:
            if reduce is None:
                stride = 1
            else:
                assert isinstance(reduce, int), 'Reduce must be an integer'
                stride = reduce

        assert (k % 2 == 1 for k in kernels), 'kernel size must be an odd number'
        self.conv2d_10 = Conv2d(in_channels=bottneck_out_channel,
                                out_channels=conv_out_channels,
                                kernel_size=kernels[0],
                                stride=stride,
                                padding=int((kernels[0]) / 2))  ### padding = kernel size//2

        self.conv2d_20 = Conv2d(in_channels=bottneck_out_channel,
                                out_channels=conv_out_channels,
                                kernel_size=kernels[1],
                                stride=stride,
                                padding=int((kernels[1]) / 2))  ### padding = kernel size//2

        self.conv2d_40 = Conv2d(in_channels=bottneck_out_channel,
                                out_channels=conv_out_channels,
                                kernel_size=kernels[2],
                                stride=stride,
                                padding=int((kernels[2]) / 2))  ### padding = kernel size//2

        #### residual_conv and bottleneck convolution must match the inputs shape [batchsize, in_channel, with, height]
        self.bottleneck = Conv2d(in_channels=in_channels, out_channels=bottneck_out_channel, kernel_size=1, stride=1)
        self.residual_conv = Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=kernels[0], stride=stride, padding=kernels[0] // 2)
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(conv_out_channels * 4)

        #
        # self.conv1d_10 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=kernels[0], stride=strides[0], padding=int((kernels[0] - 1) / 2))
        # self.conv1d_20 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=kernels[1], stride=strides[1], padding=int((kernels[1] - 1) / 2))
        # self.conv1d_40 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=kernels[2], stride=strides[2], padding=int((kernels[2] - 1) / 2))
        # #### residual_conv and bottleneck convolution must match the inputs shape [batchsize, in_channel, with, height]
        # self.bottleneck = ConvNorm(in_channels=in_channels, out_channels=bottneck_out_channel, kernel_size=1, stride=1, padding=0)
        # self.residual_conv = ConvNorm(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=1, stride=1, padding=0)
        # self.max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.batch_norm = torch.nn.BatchNorm2d(conv_out_channels)

    def forward(self, inputs):

        pool_out = self.max_pooling(inputs)
        # print(f'pool_out shape {pool_out.shape}')

        residual_out = self.residual_conv(pool_out)
        # print(f'residual_out shape {residual_out.shape}')

        bottleneck_output = self.bottleneck(inputs)
        # print(f'bottleneck_output shape {bottleneck_output.shape}')

        conv_10_out = self.conv2d_10(bottleneck_output)
        # print(f'conv_10_out shape {conv_10_out.shape}')

        conv_20_out = self.conv2d_20(bottleneck_output)
        # print(f'conv_20_out shape {conv_20_out.shape}')

        conv_40_out = self.conv2d_40(bottleneck_output)
        # print(f'conv_40_out shape {conv_40_out.shape}')

        conv_outs = torch.cat((conv_10_out, conv_20_out, conv_40_out, residual_out), dim=1)
        output = self.batch_norm(conv_outs)
        return output


class Inception(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bottnect_channels=8, n_blocks=1, reduce=4):
        '''
        Inception network
        params:
        in_channels: input channels
        bottnect_channels: bottneck channels
        conv_out_channels: intermediate convolutional channels (out channels = conv_out_channels * 4 (concat of 4 convs)
        n_blocks: number of inception blocks in this network
        reduce: output dimensions will be reduced [reduce] times.
        '''
        super(Inception, self).__init__()
        self.net = torch.nn.Sequential()
        assert out_channels % 4 == 0, 'out channels of Inception layer must be devided by 4'
        self.net.add_module(name='expand_conv', module=Conv2d(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1))  ### padding = kernel size//2)
        self.net.add_module(name='pool_first', module=nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        for i in range(n_blocks):
            if i < n_blocks - 1:
                self.net.add_module(name=f'inc_{i}', module=Inception_Block(in_channels=out_channels,
                                                                            bottneck_out_channel=bottnect_channels,
                                                                            conv_out_channels=out_channels // 4,
                                                                            kernels=None,
                                                                            stride=None,
                                                                            reduce=None))
            else:
                self.net.add_module(name=f'inc_{i}', module=Inception_Block(in_channels=out_channels,
                                                                            bottneck_out_channel=bottnect_channels,
                                                                            conv_out_channels=out_channels // 4,
                                                                            kernels=None,
                                                                            stride=None,
                                                                            reduce=None))
            self.net.add_module(name=f'pool_{i}', module=nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

        self.net.add_module(name='compress_conv', module=Conv2d(in_channels=out_channels,
                                                                out_channels=3,
                                                                kernel_size=1,
                                                                stride=1,
                                                                padding=0))  ### padding = kernel size//2)
        self.net.add_module(name=f'pool_last', module=nn.MaxPool2d(kernel_size=5, stride=4, padding=2))

    def forward(self, inputs):
        # print(f'inputs shape {inputs.shape}')
        out = self.net(inputs)
        #
        # out = self.net.inc_0(out)
        # print(f'output shape {out.shape}')
        # out = self.net.inc_1(out)
        return out


class Model(torch.nn.Module):
    def __init__(self, num_classes=80, inc_blocks=1):
        super(Model, self).__init__()
        self.dropout = nn.Dropout2d(p=0.5)
        self.inception = Inception(in_channels=3, out_channels=12, n_blocks=inc_blocks, reduce=8)
        self.flatten = torch.nn.Flatten()
        self.classify = torch.nn.Linear(in_features=6144 * 2, out_features=num_classes)
        self.relu = nn.ReLU()
        self.rep_dim = num_classes
        # self.unknown_detect = torch.nn.Sequential(torch.nn.Linear(in_features=6144, out_features=2),
        #                                           torch.nn.Sigmoid())

    def forward(self, inputs):
        extracted_feature = self.inception(inputs)
        extracted_feature = self.dropout(extracted_feature)
        # print(extracted_feature.shape)
        # feature_set_1, feature_set_2 = torch.chunk(extracted_feature,dim=1,chunks=2)
        feature_set_1 = self.flatten(extracted_feature)
        # features = torch.mean(extracted_feature, dim=(2, 3))
        features = self.dropout(feature_set_1)
        # feature_set_2 = self.flatten(feature_set_2)
        #
        # print(f'feature_set_1.shape {feature_set_1.shape}')
        # print(f'features.shape {features.shape}')
        class_out = self.classify(features)
        # class_out = self.relu(class_out)
        # unknown_out = self.unknown_detect(feature_set_2)
        return class_out


################################### training ##########################################

# import albumentations
# from albumentations.pytorch import ToTensorV2
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def mkdir(dirname1, dirname2, dirname3):
    os.makedirs(dirname1, exist_ok=True)
    os.makedirs(dirname2, exist_ok=True)
    os.makedirs(dirname3, exist_ok=True)


def init_C(model, train_loader, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        print(f'Initializing SVDD center (C)..')
        for i, sample in enumerate(train_loader):
            print(f'Remaining: {len(train_loader) - i}...')
            # get the inputs of the batch
            inputs = sample['img'].to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    print(f'Initialized SVDD center (C) successfully')
    return c


def val(val_loader, model, C, R, nu):
    model.eval()
    correct = 0
    count = 0
    SVDD_scores = []
    for i, sample in enumerate(val_loader):
        rand = randint(1, 100)
        input = sample['img'].to(device)
        target_ = sample['label'].to(device).squeeze().float()
        count += input.size(0)
        with torch.no_grad():
            class_out = model(input)

            pred_lbl_idx = torch.argmax(class_out)
            target_lbl_idx = torch.argmax(target_)
            correct += torch.sum(pred_lbl_idx == target_lbl_idx)

            #### calculate SVDD score
            dist = torch.sum((class_out - C) ** 2, dim=1)
            scores = dist
            SVDD_scores += list(scores.data.tolist())
        if rand % 100 == 0:
            print(f'pred: {pred_lbl_idx.item()} -- real: {target_lbl_idx.item()} -- distance to center: {scores.item()}')
    print(f'Validation accuracy: {correct}/{count}')
    val_acc = correct/count
    return SVDD_scores, val_acc


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def train(train_loader, model, optimizer, criterion, C, R, nu):
    model.train()
    loss_mean = 0
    local_iter = 0
    print(f'Center C: {C}')
    for i, sample in enumerate(train_loader):
        # print(f'{i}/{len(train_loader)}')
        rand = randint(0, len(train_loader))
        optimizer.zero_grad()
        input = sample['img'].to(device)
        target_ = sample['label'].to(device).squeeze().float()
        target = torch.argmax(target_, dim=1)
        class_out = model(input)

        # ### distance from predicted point to the center of SVDD
        distance = torch.sum((class_out - C) ** 2, dim=1)
        #score = distance - (R ** 2)
        svdd_loss = distance
        #svdd_loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        svdd_loss = torch.sum(svdd_loss,dim=0) / 20

        model_loss = criterion(class_out, target)
        loss = model_loss + svdd_loss
        # loss_mean += loss

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            R.data = torch.tensor(get_radius(distance, nu), device=device)
            print('')
            print(f'Radius: \n {R}')
            print(f'Iter: {i}/{len(train_loader)} -- loss : {loss.item()} -- svdd loss: {float(svdd_loss.data)} -- efficient loss: {float(model_loss.data)}')

        local_iter += 1
        # if rand % 9 == 0:
        #   print(f'pred: {class_out[0]}')
        #  print(f'real: {target_[0]}')

    # loss_mean = loss_mean / local_iter
    ### update center C
    C = init_C(model, train_loader)

    return model, C, R


def save_model(model, optimizer, epoch, path):
    ### Save model
    print(f'Saving model to {path}..')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, path)
    print('')
    print('Done!!..')




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-folder', type=str, default='checkpoints', help='path to model weights folder')
    parser.add_argument('--result-folder', type=str, default='svdd_scores', help='path to model result')

    parser.add_argument('--epoch-max', type=int, default=1200, help='max of training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from epoch')
    parser.add_argument('--batch-size', type=int, default=32, help='size of a batch')
    parser.add_argument('--num-classes', type=int, default=80, help='number of data classes')
    # parser.add_argument('--person-score', type=float, default=0.45, help='the person score threshold [0.0 ~ 1.0]')
    # parser.add_argument('--iou', type=float, default=0.45, help='the iou threshold [0.0 ~ 1.0]')
    # parser.add_argument('--img-size', type=list, action='store', default=(416, 416), help='input image size')
    # parser.add_argument('--image', default=False, action="store_true", help='image detection mode , boolean type')
    # parser.add_argument('--video', type=str, default='rtsp://itaz:12345@192.168.0.33:554/stream_ch00_0', help='path to the video')
    # parser.add_argument('--output', default=False, action="store_true", help='whether save the output to video file, boolean type')
    # parser.add_argument('--weights', nargs='+', type=str, default='models/checkpoints/yolov5s.pt', help='yolo5 checkpoint path(s)')
    # parser.add_argument('--view-img', action='store_true', help='display results, boolean type')
    # parser.add_argument('--save-img', action='store_true', help='save results , boolean type')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt , boolean type')
    # parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    epoch_max = args.epoch_max
    checkpoints_folder = args.checkpoints_folder
    checkpoint = os.path.join(checkpoints_folder, f'best_model.pth')
    result_path = args.result_folder
    start_epoch = args.start_epoch
    gpu_numb = 0
    batch_size = args.batch_size if torch.cuda.is_available() else 2
    test_batch_size = 1
    num_classes = args.num_classes

    image_save = False

    os.makedirs(checkpoints_folder, exist_ok=True)
    train_dirname = './converse/train'
    test_dirname = './converse/test'
    unknown_dirname = './converse/unknown'

    device = torch.device("cuda:%d" % gpu_numb if torch.cuda.is_available() else "cpu")
    nu = 0.1  ### Deep svdd params
    ### Load train and test set
    list_train, len_train = mklist(root_dir=train_dirname)
    list_test, len_test = mklist(root_dir=test_dirname)
    list_unknown, len_unknown = mklist(root_dir=unknown_dirname)
    random.shuffle(list_train)
    random.shuffle(list_test)
    random.shuffle(list_unknown)

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
    ood_dataset = loader.dataset(list_unknown, one_hot_encoder)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=0)
    ood_loader = torch.utils.data.DataLoader(ood_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)

    ##### load model:
    model = efficientnet_b3(pretrained=False, num_classes=80).to(device)
    # model.to(device)
    # checkpoint = torch.load('./checkpoints/' + 'best_model' + '.pth', map_location=device)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    if os.path.isfile(checkpoint):
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['epoch']
        print("=== Model is loaded")
    else:
        start_epoch = 0

    ### Define model and loss
    # model = Model(num_classes=80, inc_blocks=1)
    model.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    ### init SVDD center
    C = init_C(model, train_loader)
    R = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)  # radius R initialized with 0 by default.

    from thien_odin import write_list_to_txt

    best_acc = 0.0

    for epoch in range(start_epoch, epoch_max):
        model, C, R = train(train_loader, model, optimizer, criterion1, C, R, nu)

        svdd_scores_iod, iod_val_acc = val(val_loader=test_loader, model=model, C=C, R=R, nu=nu)
        svdd_scores_ood, _ = val(val_loader=ood_loader, model=model, C=C, R=R, nu=nu)

        result_path = f'svdd_scores_/{epoch}'
        os.makedirs(result_path, exist_ok=True)
        write_list_to_txt(svdd_scores_iod, os.path.join(result_path, "in_distribution.txt"))
        write_list_to_txt(svdd_scores_ood, os.path.join(result_path, "out_of_distribution.txt"))

        svdd_scores_iod = sorted(svdd_scores_iod)
        threshold = svdd_scores_iod[int(len(svdd_scores_iod) * 0.90)]
        print(f'threshold: {threshold} -- iod acc: {float(iod_val_acc.cpu().numpy())} -- ood acc: {sum(score > threshold for score in svdd_scores_ood)}/{len(svdd_scores_ood)}')
        ood_val_acc = sum(score > threshold for score in svdd_scores_ood)/len(svdd_scores_ood)
        # print(iod_val_acc)
        # print(ood_val_acc)
        total_acc = (float(iod_val_acc.cpu().numpy()) + float(ood_val_acc)) / 2

        if total_acc > best_acc:
            best_acc = total_acc
            save_model(model=model, optimizer=optimizer, epoch=epoch,path='checkpoints/best_model.pth')

        save_model(model=model, optimizer=optimizer, epoch=epoch, path=f'checkpoints/epoch-{epoch}_val_acc-{iod_val_acc}_ood_acc-{ood_val_acc}.pth')


