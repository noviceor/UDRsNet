import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
# Import H W
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

# Alignment Loss in Optimization of Warp section

def cal_lp_loss(input1, input2, output_H1, output_H_inv1, output_H_total, output_H_inv_total, warp_mesh, warp_mesh_mask):
    """
    Calculate the LP loss for image alignment and transformation.

    Args:
        input1 (torch.Tensor): First input image tensor with shape (batch_size, channels, height, width).
        input2 (torch.Tensor): Second input image tensor with shape (batch_size, channels, height, width).
        output_H1 (torch.Tensor): First homography transformation tensor with shape (batch_size, 6, height, width).
        output_H_inv1 (torch.Tensor): Inverse of the first homography transformation tensor with shape (batch_size, 6, height, width).
        output_H_total (torch.Tensor): Second homography transformation tensor with shape (batch_size, 6, height, width).
        output_H_inv_total (torch.Tensor): Inverse of the second homography transformation tensor with shape (batch_size, 6, height, width).
        warp_mesh (torch.Tensor): Tensor representing the warped mesh with shape (batch_size, channels, height, width).
        warp_mesh_mask (torch.Tensor): Mask tensor for the warp mesh with shape (batch_size, channels, height, width).

    Returns:
        Tensor: Calculated LP loss.
    """
    batch_size, _, img_h, img_w = input1.size()

    # Part 1: Symmetric Homography Loss with Color Balance
    delta1 = (torch.sum(output_H1[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input1 * output_H1[:, 3:6, :, :], dim=[2, 3])) \
             / torch.sum(output_H1[:, 3:6, :, :], dim=[2, 3])
    input1_balance = input1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = (torch.sum(output_H_inv1[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input2 * output_H_inv1[:, 3:6, :, :],
                                                                             dim=[2, 3])) \
             / torch.sum(output_H_inv1[:, 3:6, :, :], dim=[2, 3])
    input2_balance = input2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta1_total = (torch.sum(output_H_total[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input1 * output_H_total[:, 3:6, :, :], dim=[2, 3])) \
              / torch.sum(output_H_total[:, 3:6, :, :], dim=[2, 3])
    input1_balance_total = input1 + delta1_total.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2_total = (torch.sum(output_H_inv_total[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input2 * output_H_inv_total[:, 3:6, :, :],
                                                                              dim=[2, 3])) \
              / torch.sum(output_H_inv_total[:, 3:6, :, :], dim=[2, 3])
    input2_balance_total = input2 + delta2_total.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_1_1 = l_num_loss(input1_balance * output_H1[:, 3:6, :, :], output_H1[:, 0:3, :, :], 1) \
                + l_num_loss(input2_balance * output_H_inv1[:, 3:6, :, :], output_H_inv1[:, 0:3, :, :], 1)

    lp_loss_1_total = l_num_loss(input1_balance_total * output_H_total[:, 3:6, :, :], output_H_total[:, 0:3, :, :], 1) \
                  + l_num_loss(input2_balance_total * output_H_inv_total[:, 3:6, :, :], output_H_inv_total[:, 0:3, :, :], 1)

    lp_loss_1 =  0.8*lp_loss_1_1 +  0.2*lp_loss_1_total
    # Part 2: TPS Loss with Color Balance
    delta3 = (torch.sum(warp_mesh, dim=[2, 3]) - torch.sum(input1 * warp_mesh_mask, dim=[2, 3])) \
             / torch.sum(warp_mesh_mask, dim=[2, 3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance * warp_mesh_mask, warp_mesh, 1)

    # Final LP Loss
    lp_loss = 3.0 * lp_loss_1 + 1.0 * lp_loss_2

    return lp_loss,lp_loss_1_1,lp_loss_1_total,lp_loss_2

# Only TPS Loss

def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss =  1. * lp_loss_2

    return lp_loss
# Loss for distortion part

def inter_grid_loss(overlap, mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)


# Loss for distortion part
# intra-grid constraint

def intra_grid_loss(pts):

    max_w = 512/grid_w * 2
    max_h = 512/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss


def cal_lp_loss_homo(input1, input2, output_H1, output_H_inv1, output_H_total, output_H_inv_total):
    batch_size, _, img_h, img_w = input1.size()

    # Part 1: Symmetric Homography Loss with Color Balance
    delta1 = (torch.sum(output_H1[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input1 * output_H1[:, 3:6, :, :], dim=[2, 3])) \
             / torch.sum(output_H1[:, 3:6, :, :], dim=[2, 3])
    input1_balance = input1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = (torch.sum(output_H_inv1[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input2 * output_H_inv1[:, 3:6, :, :],
                                                                             dim=[2, 3])) \
             / torch.sum(output_H_inv1[:, 3:6, :, :], dim=[2, 3])
    input2_balance = input2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta1_total = (torch.sum(output_H_total[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input1 * output_H_total[:, 3:6, :, :], dim=[2, 3])) \
              / torch.sum(output_H_total[:, 3:6, :, :], dim=[2, 3])
    input1_balance_total = input1 + delta1_total.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2_total = (torch.sum(output_H_inv_total[:, 0:3, :, :], dim=[2, 3]) - torch.sum(input2 * output_H_inv_total[:, 3:6, :, :],
                                                                              dim=[2, 3])) \
              / torch.sum(output_H_inv_total[:, 3:6, :, :], dim=[2, 3])
    input2_balance_total = input2 + delta2_total.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_1_1 = l_num_loss(input1_balance * output_H1[:, 3:6, :, :], output_H1[:, 0:3, :, :], 1) \
                + l_num_loss(input2_balance * output_H_inv1[:, 3:6, :, :], output_H_inv1[:, 0:3, :, :], 1)

    lp_loss_1_total = l_num_loss(input1_balance_total * output_H_total[:, 3:6, :, :], output_H_total[:, 0:3, :, :], 1) \
                  + l_num_loss(input2_balance_total * output_H_inv_total[:, 3:6, :, :], output_H_inv_total[:, 0:3, :, :], 1)

    lp_loss_1 =  0.8*lp_loss_1_1 +  0.2*lp_loss_1_total
    # Final LP Loss (only Homography part)
    lp_loss = 3.0 * lp_loss_1

    return lp_loss, lp_loss_1_1, lp_loss_1_total
