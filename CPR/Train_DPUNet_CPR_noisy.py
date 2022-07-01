import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import h5py
import numpy as np
import os
import math
from torch.utils.data import Dataset, DataLoader
import platform
import random
from argparse import ArgumentParser
from function import cpr_forward, cpr_backward, complex_abs, complex_abs2
from function import adaptive_instance_normalization as adain
from torch.cuda.amp import autocast as autocast

parser = ArgumentParser(description='DPUNet')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=50, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
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

nrtrain = 160000   # number of training blocks
batch_size = 40

Training_data_Name = 'Training_Data_64_160000.mat'
Training_data = h5py.File('./%s/%s' % (args.data_dir, Training_data_Name),'r')
Training_labels = np.transpose(np.array(Training_data['labels']).astype(np.float32), [2, 1, 0])

class DyProxNet(nn.Module):
    def __init__(self):
        super(DyProxNet, self).__init__()
        in_nc = 2
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

        x_input = x_img.unsqueeze(1)
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
        x_pred = conv5_feat.squeeze(1)

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

    def forward(self, Phix, Mask, SapM, Hyperparam):

        B_size = Phix.shape[0]
        H_size = Mask.shape[0]
        W_size = Mask.shape[1]
        x = torch.ones(B_size,H_size,W_size).to(device)

        for i in range(self.LayerNo):
            z_hat = cpr_forward(x, Mask, SapM)
            Phix_hat = complex_abs(z_hat)
            meas_err = Phix_hat - Phix
            gradient_forward = torch.stack((meas_err/Phix_hat*z_hat[...,0], meas_err/Phix_hat*z_hat[...,1]), -1)
            gradient = cpr_backward(gradient_forward, Mask, SapM)
            gradient_real = gradient[...,0]
            x = x - self.step_size[i] * gradient_real
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
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler() 
model_dir = "./%s/CPR_noisy_DPUNet_layer_%d_group_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, learning_rate)

log_file_name = "./%s/Log_CPR_noisy_DPUNet_layer_%d_group_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, learning_rate)

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

        sampling_rate = random.choice([30, 40, 50])
        Height_x = batch_x.shape[1]
        Weight_x = batch_x.shape[2]
        n_output = Height_x*Weight_x
        n_input = n_output*sampling_rate/100

        # Load CPR SampM: SampM
        SampM_data_Name = './%s/SampM_%d_64.mat' % (args.matrix_dir,sampling_rate)
        SampM_data = sio.loadmat(SampM_data_Name)
        SampM_input = SampM_data['SubsampM']
        SampM = torch.from_numpy(SampM_input).type(torch.FloatTensor).to(device)


        # Load CPR mask: mask
        Mask_data_Name = './%s/mask_0_64.mat' % (args.matrix_dir)
        if os.path.exists(Mask_data_Name):
            Mask_data = sio.loadmat(Mask_data_Name)
            Mask_input = Mask_data['Mask']
        else:
            Mask_input = np.exp(1j*2*np.pi*np.random.rand(Height_x, Weight_x))
            sio.savemat(Mask_data_Name,{'Mask':Mask_input})
        Mask_input = np.stack((Mask_input.real, Mask_input.imag), axis=-1)
        Mask = torch.from_numpy(Mask_input).type(torch.FloatTensor).to(device)

        z = cpr_forward(batch_x, Mask, SampM)
        Phix = complex_abs2(z)
        Phix_sqrt = complex_abs(z)

        # Add Poisson noise
        alpha = torch.FloatTensor([np.random.uniform(0, 30)]).to(device)
        noise = alpha/255 * Phix_sqrt*torch.randn(Phix.size()).float().to(device)

        Rawdata = Phix + noise
        Y = torch.clamp(Rawdata, min=0.0)
        y = torch.sqrt(Y)

        Hypern = torch.FloatTensor([sampling_rate/50,alpha/50]).to(device)

        # Zero gradients
        optimizer.zero_grad()

        #### use amp to accelerate training process
        if use_amp:
            with autocast():
                x_output = model(y, Mask, SampM, Hypern)

                # Compute MEM
                Phix_hat =  complex_abs(cpr_forward(x_output, Mask, SampM))
                loss_measurement = torch.mean(torch.pow(Phix_hat - Phix_sqrt, 2))

                # Compute MSE
                loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
                
                loss_all = loss_discrepancy


            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            x_output = model(y, Mask, SampM, Hypern)

            # Compute MEM
            Phix_hat =  complex_abs(cpr_forward(x_output, Mask, SampM))
            loss_measurement = torch.mean(torch.pow(Phix_hat - Phix_sqrt, 2))

            # Compute MSE
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            
            loss_all = loss_discrepancy
            # perform a backward pass, and update the weights.
            loss_all.backward()
            optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.4f, Measurement Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_measurement.item() )
        print(output_data)
    
    # scheduler.step()
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
        print("The learning rate of %d epoch: %f" % (epoch_i, optimizer.param_groups[0]['lr']))