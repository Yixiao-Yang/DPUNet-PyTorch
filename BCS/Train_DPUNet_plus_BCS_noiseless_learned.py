import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import random
from utilities import adaptive_instance_normalization as adain
from utilities import img2col_batch_py, col2im_CS_batch_py
from archs import UNet_dynamic
from torch.cuda.amp import autocast as autocast

parser = ArgumentParser(description='DPUNet-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=20, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--use_amp', type=str, default='True', help='use amp for training')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--save_interval', type=int, default=1, help='interval of saving model')


args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list
use_amp = args.use_amp


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_output = 1089
nrtrain = 80000   # number of training blocks
batch_size = 30

Training_data_Name = 'Training_Data_99.mat'
Training_data = h5py.File('./%s/%s' % (args.data_dir, Training_data_Name),'r')
Training_labels = np.transpose(np.array(Training_data['labels']).astype(np.float32), [2, 1, 0])

class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply

class DyProxNet(nn.Module):
    def __init__(self):
        super(DyProxNet, self).__init__()
        in_nc = 1
        feature1 = 576
        feature2 = 36864
        feature3 = 64

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_nc,feature1)
        self.fc2 = nn.Linear(in_nc,feature2)
        self.fc3 = nn.Linear(in_nc,feature2)
        self.fc4 = nn.Linear(in_nc,feature2)
        self.fc5 = nn.Linear(in_nc,feature1)
        
        self.eta1 = nn.Linear(in_nc,feature3)
        self.eta2 = nn.Linear(in_nc,feature3)
        self.eta3 = nn.Linear(in_nc,feature3)
        self.eta4 = nn.Linear(in_nc,feature3)
        self.beta1 = nn.Linear(in_nc,feature3)
        self.beta2 = nn.Linear(in_nc,feature3)
        self.beta3 = nn.Linear(in_nc,feature3)
        self.beta4 = nn.Linear(in_nc,feature3)

        self.eta11 = nn.Linear(feature3,feature3)
        self.eta21 = nn.Linear(feature3,feature3)
        self.eta31 = nn.Linear(feature3,feature3)
        self.eta41 = nn.Linear(feature3,feature3)
        self.beta11 = nn.Linear(feature3,feature3)
        self.beta21 = nn.Linear(feature3,feature3)
        self.beta31 = nn.Linear(feature3,feature3)
        self.beta41 = nn.Linear(feature3,feature3)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)


    def forward(self, x_param, x_img):

        x_input = x_img.clone()
        conv1_weight = self.fc1(x_param)
        conv2_weight = self.fc2(x_param)
        conv3_weight = self.fc3(x_param)
        conv4_weight = self.fc4(x_param)
        conv5_weight = self.fc5(x_param)

        eta1 = self.eta11(self.relu(self.eta1(x_param))).reshape(1,64,1,1)
        eta2 = self.eta21(self.relu(self.eta2(x_param))).reshape(1,64,1,1)
        eta3 = self.eta31(self.relu(self.eta3(x_param))).reshape(1,64,1,1)
        eta4 = self.eta41(self.relu(self.eta4(x_param))).reshape(1,64,1,1)
        beta1 = self.beta11(self.relu(self.beta1(x_param))).reshape(1,64,1,1)
        beta2 = self.beta21(self.relu(self.beta2(x_param))).reshape(1,64,1,1)
        beta3 = self.beta31(self.relu(self.beta3(x_param))).reshape(1,64,1,1)
        beta4 = self.beta41(self.relu(self.beta4(x_param))).reshape(1,64,1,1)

        conv1_feat = self.relu(adain(F.conv2d(x_input, conv1_weight.reshape(64,1,3,3), bias=None, padding=1),eta1,beta1))
        conv2_feat = self.relu(adain(F.conv2d(conv1_feat, conv2_weight.reshape(64,64,3,3), bias=None, padding=1),eta2,beta2))
        conv3_feat = self.relu(adain(F.conv2d(conv2_feat, conv3_weight.reshape(64,64,3,3), bias=None, padding=1),eta3,beta3))
        conv4_feat = self.relu(adain(F.conv2d(conv3_feat, conv4_weight.reshape(64,64,3,3), bias=None, padding=1),eta4,beta4))
        conv5_feat = F.conv2d(conv4_feat, conv5_weight.reshape(1,64,3,3), bias=None, padding=1)
        x_pred = conv5_feat.clone()

        return x_pred + x_img


# Define DPUNet
class DPUNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(DPUNet, self).__init__()

        self.step_size = nn.Parameter(0.5*torch.ones(LayerNo))
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(545, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
   
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(DyProxNet())

        self.fcs = nn.ModuleList(onelayer)

        self.deblock = UNet_dynamic(n_channels=1, n_classes=1)

    def forward(self, batch_x, n_input, Hyperparam):
        
        input_x = img2col_batch_py(batch_x, 33)

        # Sampling-subnet
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_[0:n_input, :]
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(input_x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements
        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization

        for i in range(self.LayerNo):
            x = x - self.step_size[i] * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
            x = x + self.step_size[i] * PhiTb
            x = self.fcs[i](Hyperparam, x)
        
        x_rec = col2im_CS_batch_py(x, 99, 99)   # Transfer to the whole patch-based reconstruction
        
        x_deblock = self.deblock(x_rec, Hyperparam)  # Deblocking

        x_final = x_rec + x_deblock
        
        return [x_final, x_rec, Phi]


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)


model = DPUNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./model/BCS_noiseless_DALM_DPUNet_layer_10_group_1_lr_0.0001/net_params_200.pkl'), strict=False)
# for name, Parameters in model.named_parameters():
#     print(name,':',Parameters)
    # import ipdb;ipdb.set_trace()

print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=8,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=8,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/BCS_DPUNet_plus_noiseless_DALM_Unet_dynamic_joint_layer_%d_group_%d" % (args.model_dir, layer_num, group_num)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)

scaler = torch.cuda.amp.GradScaler() 

log_file_name = "./%s/Log_BCS_DPUNet_plus_noiseless_DALM_Unet_dynamic_joint_layer_%d_group_%d.txt" % (args.log_dir, layer_num, group_num)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    for data in rand_loader:

        batch_x = data.view(-1, 1, 99, 99)
        
        batch_x = batch_x.to(device)

        cs_ratio = random.choice([1, 4, 10, 25, 40, 50])

        ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

        n_input = ratio_dict[cs_ratio]

        Eye_I = torch.eye(n_input).to(device)

        Hypern = torch.FloatTensor([cs_ratio/50]).to(device)

        # Zero gradients
        optimizer.zero_grad()

        #### use amp to accelerate training process
        if use_amp:
            with autocast():

                [x_output, x_rec, Phi] = model(batch_x, n_input, Hypern)
                
                # Compute and print loss
                loss_pre = torch.mean(torch.pow(x_rec - batch_x, 2))
                
                loss_orth = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1))-Eye_I, 2))
                
                mu = torch.Tensor([0.01]).to(device)
                
                loss_deblock = torch.mean(torch.pow(x_output - batch_x, 2))

                loss_all = loss_deblock + torch.mul(mu, loss_orth)

            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            [x_output, x_rec, Phi] = model(batch_x, n_input, Hypern)
                
            # Compute and print loss
            loss_pre = torch.mean(torch.pow(x_rec - batch_x, 2))
            
            loss_orth = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1))-Eye_I, 2))
            
            mu = torch.Tensor([0.01]).to(device)
            
            loss_deblock = torch.mean(torch.pow(x_output - batch_x, 2))

            loss_all = loss_deblock + torch.mul(mu, loss_orth)
            # perform a backward pass, and update the weights.
            loss_all.backward()
            optimizer.step()

        output_data = "[%02d/%02d] All Loss: %.4f, Deblock Loss: %.4f, Preconstruction Loss: %.4f, Ortho Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_deblock.item(), loss_pre.item(), loss_orth.item())
        print(output_data)

    # scheduler.step()
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
