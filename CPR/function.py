import torch


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
    return normalized_feat * eta.expand(size) + beta.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

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
    masked_data = torch.stack((data*Mask_real, data*Mask_imag), -1)
    fourier_data = torch.fft(masked_data, 2, normalized=True)
    # import ipdb; ipdb.set_trace()
    forward_data = torch.stack((torch.mm(fourier_data[...,0].reshape(B_size,Hight*Width),samplematrix.transpose(0,1)), torch.mm(fourier_data[...,-1].reshape(B_size,Hight*Width),samplematrix.transpose(0,1))), -1)
    
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
    back_data = torch.stack((torch.mm(data[...,0],samplematrix), torch.mm(data[...,-1],samplematrix)), -1)
    Ifft_data = torch.ifft(back_data.reshape(batch_size,hight,width,2), 2, normalized=True)
    
    Mask_real = torch.unsqueeze(mask[...,0], 0)
    Mask_imag = torch.unsqueeze(mask[...,-1], 0)
    backward_data = torch.stack((Mask_real*Ifft_data[...,0]+Mask_imag*Ifft_data[...,-1], Mask_real*Ifft_data[...,-1]-Mask_imag*Ifft_data[...,0]), -1)
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
