import h5py
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random
from argparse import ArgumentParser
from utilities import adaptive_instance_normalization as adain
from torch.cuda.amp import autocast as autocast
import types

parser = ArgumentParser(description='DPUNet-MIX')

parser.add_argument('--start_step', type=int, default=0, help='iteration number of start training')
parser.add_argument('--end_step', type=int, default=200000, help='iteration number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet-MIX')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=2, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')
parser.add_argument('--use_amp', type=str, default='True', help='use amp for training')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_step = args.start_step
end_step = args.end_step
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list
use_amp = args.use_amp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_output1 = 1089
nrtrain1 = 88912   # number of training blocks
batch_size1 = 64

Training_data_Name1 = 'Training_Data.mat'
Training_data1 = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name1))
Training_labels1 = Training_data1['labels']

nrtrain2 = 800   # number of training blocks
batch_size2 = 4

Training_data_Name2 = 'Training_BrainImages_256x256_100.mat'
Training_data2 = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name2))
Training_labels2 = Training_data2['labels']

nrtrain3 = 160000   # number of training blocks
batch_size3 = 40

Training_data_Name3 = 'Training_Data_64_160000.mat'
Training_data3 = h5py.File('./%s/%s' % (args.data_dir, Training_data_Name3),'r')
Training_labels3 = np.transpose(np.array(Training_data3['labels']).astype(np.float32), [2, 1, 0])
    
if isinstance(torch.fft, types.ModuleType):
    from utilities import cpr_forward, cpr_backward, complex_abs, complex_abs2
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, mask):
            x_dim_0 = x.shape[0]
            x_dim_1 = x.shape[1]
            x_dim_2 = x.shape[2]
            x_dim_3 = x.shape[3]
            x = x.view(-1, x_dim_2, x_dim_3, 1)
            y = torch.zeros_like(x)
            z = torch.complex(x, y)
            fftz = torch.fft.fft2(z, dim=(1,2), norm="ortho")
            mask = torch.complex(mask[...,0],mask[...,1]).unsqueeze(-1)
            z_hat = torch.fft.ifft2(fftz * mask, dim=(1,2), norm="ortho")
            x = torch.real(z_hat)
            x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
            return x

else:
    from function import cpr_forward, cpr_backward, complex_abs, complex_abs2
    class FFT_Mask_ForBack(torch.nn.Module):
        def __init__(self):
            super(FFT_Mask_ForBack, self).__init__()

        def forward(self, x, mask):
            x_dim_0 = x.shape[0]
            x_dim_1 = x.shape[1]
            x_dim_2 = x.shape[2]
            x_dim_3 = x.shape[3]
            x = x.view(-1, x_dim_2, x_dim_3, 1)
            y = torch.zeros_like(x)
            z = torch.cat([x, y], 3)
            fftz = torch.fft(z, 2)
            z_hat = torch.ifft(fftz * mask, 2)
            x = z_hat[:, :, :, 0:1]
            x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
            return x

class DyProxNet(nn.Module):
    def __init__(self):
        super(DyProxNet, self).__init__()
        in_nc = 3
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

        x_input = x_img
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
        x_pred = conv5_feat

        return x_pred + x_img


# Define DPUNet
class DPUNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(DPUNet, self).__init__()

        self.step_size = nn.Parameter(0.5*torch.ones(LayerNo))
        self.fft_forback = FFT_Mask_ForBack()
   
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(DyProxNet())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, Mask, SampM, Hyperparam):

        task_ind = Hyperparam[0]
        # initialization
        if task_ind == 1/5:
            # CS initialization
            PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
            PhiTb = torch.mm(Phix, Phi)
            x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        if task_ind == 2/5:
            # CS-MRI initialization
            x = Phix
        
        if task_ind == 3/5:
            # CPR initialization
            B_size = Phix.shape[0]
            H_size = Mask.shape[0]
            W_size = Mask.shape[1]
            x = torch.ones(B_size,1,H_size,W_size).to(device)

        for i in range(self.LayerNo):
            if task_ind == 1/5:
                # CS gradient descent
                x = x.view(-1, 1089)
                x = x - self.step_size[i] * torch.mm(x, PhiTPhi)
                x = x + self.step_size[i] * PhiTb
                x = x.view(-1, 1, 33, 33)

            if task_ind == 2/5:
                # CS-MRI gradient descent
                # x = x.unsqueeze(1)
                x = x - self.step_size[i] * self.fft_forback(x, Mask)
                x = x + self.step_size[i] * Phix
                # x = x.squeeze(1)
            
            if task_ind == 3/5:
                # CPR gradient descent
                x = x.squeeze(1)
                z_hat = cpr_forward(x, Mask, SampM)
                Phix_hat = complex_abs(z_hat)
                meas_err = Phix_hat - Phix
                gradient_forward = torch.stack((meas_err/Phix_hat*z_hat[...,0], meas_err/Phix_hat*z_hat[...,1]), -1)
                gradient = cpr_backward(gradient_forward, Mask, SampM)
                gradient_real = gradient[...,0]
                x = x - self.step_size[i] * gradient_real
                x = x.unsqueeze(1)

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


rand_loader1 = DataLoader(dataset=RandomDataset(Training_labels1, nrtrain1), batch_size=batch_size1, num_workers=0,
                            shuffle=True)
rand_loader2 = DataLoader(dataset=RandomDataset(Training_labels2, nrtrain2), batch_size=batch_size2, num_workers=0,
                            shuffle=True)
rand_loader3 = DataLoader(dataset=RandomDataset(Training_labels3, nrtrain3), batch_size=batch_size3, num_workers=0,
                            shuffle=True)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scaler = torch.cuda.amp.GradScaler() 

model_dir = "./%s/MIX_noisy_DPUNet_layer_%d_group_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, learning_rate)

log_file_name = "./%s/Log_MIX_noisy_DPUNet_layer_%d_group_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_step > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_step)))

# Training loop
step_i = start_step
data_iterator1 = iter(rand_loader1)
data_iterator2 = iter(rand_loader2)
data_iterator3 = iter(rand_loader3)
while step_i < end_step:
    if step_i % 3 == 0:
        task_ind = 1
        try:
            data = data_iterator1.next()
        except StopIteration:
            data_iterator1 = iter(rand_loader1)
            data = data_iterator1.next()
            
    if step_i % 3 == 1:
        task_ind = 2
        try:
            data = data_iterator2.next()
        except StopIteration:
            data_iterator2 = iter(rand_loader2)
            data = data_iterator2.next()       

    if step_i % 3 == 2:
        task_ind = 3
        try:
            data = data_iterator3.next()
        except StopIteration:
            data_iterator3 = iter(rand_loader3)
            data = data_iterator3.next()    
    
    batch_x = data.to(device)
    step_i += 1
    # others are the same
           
    # choose excute the task: 1 denotes block-based compressive sensing; 2 denotes CS-MRI; 3 denotes compressive phase retrieval
    # task_ind = random.choice([1,2])
    if task_ind == 1:
        # generate CS measurements           
        cs_ratio = random.choice([1, 4, 10, 25, 40, 50])
        im_conditions = cs_ratio/50
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
            X_data = Training_labels1.transpose()
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

        Mask = torch.FloatTensor([0]).to(device)
        SampM = torch.FloatTensor([0]).to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        # Add Gaussian noise
        alpha = torch.FloatTensor([np.random.uniform(0, 50)]).to(device)
        noise = alpha/255 * torch.randn(Phix.size()).float().to(device)
        y = Phix + noise

    if task_ind == 2:

        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        

        cs_ratio = random.choice([20, 30, 40, 50])
        
        # Load CS Sampling Matrix: phi
        Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
        Phi_data = sio.loadmat(Phi_data_Name)
        mask_matrix = Phi_data['mask_matrix']
        mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
        mask = torch.unsqueeze(mask_matrix, 2)
        Mask = torch.cat([mask, mask], 2)
        Mask = Mask.to(device)

        Phi = torch.FloatTensor([0]).to(device)
        Qinit = torch.FloatTensor([0]).to(device)
        SampM = torch.FloatTensor([0]).to(device)

        PhiTb = FFT_Mask_ForBack()(batch_x, Mask)

        # Add Gaussian noise
        alpha = torch.FloatTensor([np.random.uniform(0, 50)]).to(device)
        noise = alpha/255 * torch.randn(PhiTb.size()).float().to(device)
        y = PhiTb + noise

    if task_ind == 3:

        cs_ratio = random.choice([30, 40, 50])
        Height_x = batch_x.shape[1]
        Weight_x = batch_x.shape[2]
        n_output = Height_x*Weight_x
        n_input = n_output*cs_ratio/100

        # Load CPR SampM: SampM
        SampM_data_Name = './%s/SampM_%d_64.mat' % (args.matrix_dir,cs_ratio)
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

        Phi = torch.FloatTensor([0]).to(device)
        Qinit = torch.FloatTensor([0]).to(device)

        z = cpr_forward(batch_x, Mask, SampM)
        Phix = complex_abs2(z)
        Phix_sqrt = complex_abs(z)

        # Add Poisson noise
        alpha = torch.FloatTensor([np.random.uniform(0, 30)]).to(device)
        noise = alpha/255 * Phix_sqrt*torch.randn(Phix.size()).float().to(device)
        
        Rawdata = Phix + noise
        Y = torch.clamp(Rawdata, min=0.0)
        y = torch.sqrt(Y)


    Hypern = torch.FloatTensor([task_ind/5, cs_ratio/50, alpha/50]).to(device)  

    # Zero gradients
    optimizer.zero_grad()

    #### use amp to accelerate training process
    if use_amp:
        with autocast():
            x_output = model(y, Phi, Qinit, Mask, SampM, Hypern)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.pow(x_output.reshape(batch_x.shape) - batch_x, 2))
            
            loss_all = loss_discrepancy


        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()

    else:
        x_output = model(y, Phi, Qinit, Mask, SampM, Hypern)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output.reshape(batch_x.shape) - batch_x, 2))

        loss_all = loss_discrepancy

        # Zero gradients, perform a backward pass, and update the weights.
        loss_all.backward()
        optimizer.step()


    output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f\n" % (step_i, end_step, loss_all.item(), loss_discrepancy.item())
    print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if step_i % 10000 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, step_i))  # save only the parameters
