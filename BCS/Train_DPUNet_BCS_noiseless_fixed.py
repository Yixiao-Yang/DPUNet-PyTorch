import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
import random
from argparse import ArgumentParser
from utilities import adaptive_instance_normalization as adain
from torch.cuda.amp import autocast as autocast

parser = ArgumentParser(description='DPUNet')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--use_amp', type=str, default='True', help='use amp for training')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

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
nrtrain = 88912   # number of training blocks
batch_size = 64

Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']

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

        x_input = x_img.view(-1, 1, 33, 33)
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
        x_pred = conv5_feat.view(-1, 1089)

        return x_pred + x_img


# Define DPUNet
class DPUNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(DPUNet, self).__init__()

        self.step_size = nn.Parameter(0.5*torch.ones(LayerNo))
   
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(DyProxNet())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, Hyperparam):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        for i in range(self.LayerNo):
            x = x - self.step_size[i] * torch.mm(x, PhiTPhi)
            x = x + self.step_size[i] * PhiTb
            x = self.fcs[i](Hyperparam, x)

        x_final = x

        return x_final

model = DPUNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

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
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scaler = torch.cuda.amp.GradScaler() 

model_dir = "./%s/BCS_noiseless_fixed_DPUNet_layer_%d_group_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, learning_rate)

log_file_name = "./%s/Log_BCS_noiseless_fixed_DPUNet_layer_%d_group_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    for data in rand_loader:

        batch_x = data
        batch_x = batch_x.to(device)

        cs_ratio = random.choice([1, 4, 10, 25, 40, 50])

        ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

        n_input = ratio_dict[cs_ratio]

        # Load CS Sampling Matrix: phi
        Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
        Phi_data = sio.loadmat(Phi_data_Name)
        Phi_input = Phi_data['phi']

        Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)

        # Computing Initialization Matrix:
        if os.path.exists(Qinit_Name):
            Qinit_data = sio.loadmat(Qinit_Name)
            Qinit = Qinit_data['Qinit']

        else:
            X_data = Training_labels.transpose()
            Y_data = np.dot(Phi_input, X_data)
            Y_YT = np.dot(Y_data, Y_data.transpose())
            X_YT = np.dot(X_data, Y_data.transpose())
            Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
            del X_data, Y_data, X_YT, Y_YT
            sio.savemat(Qinit_Name, {'Qinit': Qinit})

        
        Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
        Phi = Phi.to(device)

        Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
        Qinit = Qinit.to(device)


        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        Hypern = torch.FloatTensor([cs_ratio/50]).to(device)   

        # Zero gradients
        optimizer.zero_grad()

        #### use amp to accelerate training process
        if use_amp:
            with autocast():
                x_output = model(Phix, Phi, Qinit, Hypern)

                # Compute and print loss
                loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                loss_all = loss_discrepancy


            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            x_output = model(Phix, Phi, Qinit, Hypern)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

            loss_all = loss_discrepancy

            # Zero gradients, perform a backward pass, and update the weights.
            loss_all.backward()
            optimizer.step()


        output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
