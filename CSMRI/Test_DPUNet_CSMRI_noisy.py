import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
from function import adaptive_instance_normalization as adain

parser = ArgumentParser(description='DPUNet')

parser.add_argument('--epoch_num', type=int, default=200, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of DPUNet')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=20, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--noise_level', type=int, default=10, help='Gaussian noise level')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Brainimages_test', help='name of test set')

args = parser.parse_args()


epoch_num = args.epoch_num
learning_rate = args.learning_rate
noise_level = args.noise_level
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']


mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)


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

    def forward(self, PhiTb, mask, Hyperparam):

        x = PhiTb

        for i in range(self.LayerNo):
            x = x - self.step_size[i] * self.fft_forback(x, mask)
            x = x + self.step_size[i] * PhiTb
            x = self.fcs[i](Hyperparam, x)

        x_final = x

        return x_final

model = DPUNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CSMRI_noisy_DPUNet_layer_%d_group_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, learning_rate)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


print('\n')
print("MRI CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Iorg = cv2.imread(imgName, 0)

        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        Img_output = Icol

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        PhiTb = FFT_Mask_ForBack()(batch_x, mask)

        # Add Gaussian noise
        alpha = torch.FloatTensor([noise_level]).to(device)
        noise = alpha/255 * torch.randn(PhiTb.size()).float().to(device)
        PhiTb = PhiTb + noise

        Hypern = torch.FloatTensor([cs_ratio/50,alpha/50]).to(device)  
        x_output = model(PhiTb, mask, Hypern)

        end = time()

        initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)

        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)

        X_init = np.clip(initial_result, 0, 1).astype(np.float64)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
        init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)

        rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)


        print("[%02d/%02d] Run time for %s is %.4f, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), init_PSNR, init_SSIM))
        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)

        resultName = imgName.replace(args.data_dir, args.result_dir)
        # cv2.imwrite("%s_CSMRI_noisy_DPUNet_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        Init_PSNR_All[0, img_no] = init_PSNR
        Init_SSIM_All[0, img_no] = init_SSIM

print('\n')
init_data =   "CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f/%.4f" % (cs_ratio, args.test_name, np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(init_data)
print(output_data)


print("MRI CS Reconstruction End")