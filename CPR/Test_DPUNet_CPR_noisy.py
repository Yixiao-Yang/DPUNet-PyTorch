import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from time import time
import math
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
from function import cpr_forward, cpr_backward, complex_abs, complex_abs2
from function import adaptive_instance_normalization as adain

parser = ArgumentParser(description='DPUNet')

parser.add_argument('--epoch_num', type=int, default=20, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--sampling_rate', type=int, default=30, help='from {30, 40, 50}')
parser.add_argument('--noise_level', type=int, default=30, help='Poisson noise level')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()


epoch_num = args.epoch_num
learning_rate = args.learning_rate
noise_level = args.noise_level
layer_num = args.layer_num
group_num = args.group_num
sampling_rate = args.sampling_rate
gpu_list = args.gpu_list
test_name = args.test_name


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nrtest = 12
batch_size = 1

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

# Define UPRNet
class UPRNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UPRNet, self).__init__()

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


model = UPRNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CPR_noisy_DPUNet_layer_%d_group_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, learning_rate)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))
model.eval()

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

PSNR_All = np.zeros([1, nrtest], dtype=np.float32)
SSIM_All = np.zeros([1, nrtest], dtype=np.float32)

Testing_data_Name = 'Test_prdeep_128.mat'
Testing_data = sio.loadmat('./%s/%s' % (args.data_dir, Testing_data_Name))
Testing_labels = Testing_data['labels']


Prediction_all = np.zeros([nrtest,128,128], dtype=np.float64)
Time_all = np.zeros([1,nrtest], dtype=np.float32)
print('\n')
print("PR Reconstruction Start")

with torch.no_grad():

    for i in range(nrtest):
        start = time()
         
        batch_x = torch.Tensor(Testing_labels[i, :, :]).unsqueeze(0).float()
        batch_x = batch_x.to(device)
        Height_x = batch_x.shape[1]
        Weight_x = batch_x.shape[2]
        n_output = Height_x*Weight_x
        n_input = n_output*sampling_rate/100

        # Load CPR SampM: SampM
        SampM_data_Name = './%s/SampM_%d_128.mat' % (args.matrix_dir,sampling_rate)
        SampM_data = sio.loadmat(SampM_data_Name)
        SampM_input = SampM_data['SubsampM']
        SampM = torch.from_numpy(SampM_input).type(torch.FloatTensor).to(device)

        # Load CPR mask: mask
        Mask_data_Name = './%s/mask_0_128.mat' % (args.matrix_dir)
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
        alpha = torch.FloatTensor([noise_level]).to(device)
        noise = alpha/255 * Phix_sqrt*torch.randn(Phix.size()).float().to(device)
        
        Rawdata = Phix + noise
        Y = torch.clamp(Rawdata, min=0.0)
        y = torch.sqrt(Y)
        
        Hypern = torch.FloatTensor([sampling_rate/50,alpha/50]).to(device)   
        x_output = model(y, Mask, SampM, Hypern)
    
        end = time()

        loss_mse = torch.mean(torch.pow(x_output - batch_x, 2))

        Prediction_value = x_output.cpu().data.numpy()
        rec_PSNR = psnr(Prediction_value.squeeze(0)*255, Testing_labels[i, :, :]*255)
        rec_SSIM = ssim(Prediction_value.squeeze(0)*255, Testing_labels[i, :]*255, data_range=255)

        print("Run time is %.4f, MSE is %.2f, PSNR is %.2f" % ((end - start), loss_mse, rec_PSNR))
        PSNR_All[0,i] = rec_PSNR
        SSIM_All[0,i] = rec_SSIM
        Prediction_all[i,:,:] = Prediction_value
        Time_all[0,i] = end - start
        

    print(np.mean(PSNR_All))
    print(np.mean(SSIM_All))
    print(np.mean(Time_all))
print("CPR Reconstruction End")