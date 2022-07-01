import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, eta, beta):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    # import ipdb;ipdb.set_trace()
    return normalized_feat * eta.expand(size) + beta.expand(size)

def cpr_forward(data, mask, samplematrix):
    """
    Compute the forward model of compressive phase retrieval.

    Args:
        data (torch.Tensor): Image_data (batch_size*hight*weight).
        mask (torch.Tensor): mask (hight*weight*2), where the size of the final dimension
            should be 2 (complex value).
        samplematrix (torch.Tensor): undersampling matrix (m*n), n = hight*weight, m = samplingratio*n

    Returns:
        forward_data (torch.Tensor): the complex field of forward data (batch_size*m*2)
    """
    
    assert mask.size(-1) == 2
    B_size = data.shape[0]
    Hight = data.shape[1]
    Width = data.shape[2]
    m = samplematrix.shape[0]
    n = samplematrix.shape[1]
    Mask_real = torch.unsqueeze(mask[...,0], 0)
    Mask_imag = torch.unsqueeze(mask[...,-1], 0)
    masked_data = torch.complex(data*Mask_real, data*Mask_imag)
    fourier_data = torch.fft.fft2(masked_data, dim=(1,2), norm="ortho")
    forward_data = torch.stack((torch.mm(torch.real(fourier_data).reshape(B_size,Hight*Width),samplematrix.transpose(0,1)), torch.mm(torch.imag(fourier_data).reshape(B_size,Hight*Width),samplematrix.transpose(0,1))), -1)
    
    return forward_data*torch.FloatTensor([n/m]).sqrt().cuda()

def cpr_backward(data, mask, samplematrix):
    """
    Compute the backward model of cpr (the inverse operator of forward model).

    Args:
        data (torch.Tensor): Field_data (batch_size*m*2).
        mask (torch.Tensor): mask (hight*width*2), where the size of the final dimension
            should be 2 (complex value).
        samplematrix (torch.Tensor): undersampling matrix (m*n).

    Returns:
        backward_data (torch.Tensor): the complex field of backward data (batch_size*hight*weight*2)
    """
    assert mask.size(-1) == 2
    batch_size = data.shape[0]
    hight = mask.shape[0]
    width = mask.shape[1]
    m = samplematrix.shape[0]
    n = samplematrix.shape[1]
    back_data = torch.complex(torch.mm(data[...,0],samplematrix).float(), torch.mm(data[...,-1],samplematrix).float())
    Ifft_data = torch.fft.ifft2(back_data.reshape(batch_size,hight,width), dim=(1,2), norm="ortho")
    
    Mask_real = torch.unsqueeze(mask[...,0], 0)
    Mask_imag = torch.unsqueeze(mask[...,-1], 0)
    backward_data = torch.stack((Mask_real * torch.real(Ifft_data)+Mask_imag * torch.imag(Ifft_data), Mask_real * torch.imag(Ifft_data)-Mask_imag * torch.real(Ifft_data)), -1)
    return backward_data*torch.FloatTensor([n/m]).sqrt().cuda()
    
def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

def complex_abs2(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)

def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col

def img2col_batch_py(Ipad, block_size):
    [batch, channel, row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = Ipad.view(batch*block_num, channel, block_size, block_size).clone()
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[count*batch:(count+1)*batch,:,:,:] = Ipad[:,:,x:x+block_size, y:y+block_size]
            count = count + 1
    
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

def col2im_CS_batch_py(X_col, row_new, col_new):
    [batch_new, channel, block_size, block_size] = X_col.shape
    row_block = row_new/block_size
    col_block = col_new/block_size
    block_num = int(row_block*col_block)
    batch = int(batch_new/block_num)
    X0_rec = X_col.view(batch, channel, row_new, col_new).clone()
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[:,:,x:x+block_size, y:y+block_size] = X_col[count*batch:(count+1)*batch,:,:,:]
            count = count + 1

    return X0_rec

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))