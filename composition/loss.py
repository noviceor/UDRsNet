import torch
import torch.nn as nn
import torch.nn.functional as F

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

def cal_pixel_term(warp1_tensor, warp2_tensor, stitched_image, overlap):
    """
    Calculate the pixel consistency loss for overlapping region.
    """
    diff_A = l_num_loss(stitched_image,warp1_tensor,1)
    diff_B = l_num_loss(stitched_image,warp2_tensor,1)
    pixel_loss = torch.min(diff_A, diff_B)
    loss = torch.mean(pixel_loss * overlap)
    return loss


def boundary_extraction(mask):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    #define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x = F.conv2d(1-mask,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)

    return x*mask
# Lc boundary
def cal_boundary_term(inpu1_tesnor, inpu2_tesnor, mask1_tesnor, mask2_tesnor, stitched_image):
    boundary_mask1 = mask1_tesnor * boundary_extraction(mask2_tesnor)
    boundary_mask2 = mask2_tesnor * boundary_extraction(mask1_tesnor)
    # boundary_mask1:Mbr  inpu1_tesnor:Iwr
    #(S − Iwr ) · Mbr
    loss1 = l_num_loss(inpu1_tesnor*boundary_mask1, stitched_image*boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tesnor*boundary_mask2, stitched_image*boundary_mask2, 1)

    return loss1+loss2, boundary_mask1

# L smooth
def cal_smooth_term_stitch(stitched_image, learned_mask1):


    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*delta,:] - stitched_image[:,:,delta:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*delta] - stitched_image[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss



def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap

    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(diff_feature[:,:,0:-1*delta,:] + diff_feature[:,:,delta:,:])
    dw_diff_img = torch.abs(diff_feature[:,:,:,0:-1*delta] + diff_feature[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss
