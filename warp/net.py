import torch
import torch.nn as nn
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import utils.torch_DLT as torch_DLT
import math
import torchvision.transforms as T
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import grid_res
from FeatureExtractor import  DAFENet
resize_512 = T.Resize((512,512))
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

def draw_mesh_on_warp(warp, f_local):
    warp = np.ascontiguousarray(warp)
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 8
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
    return warp

def H2Mesh(H, rigid_mesh):
    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()
    ori_pt = torch.cat((ori_pt, ones), 2)
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1))
    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])
    return mesh

def get_rigid_mesh(batch_size, height, width):
    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2)
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)
    return norm_mesh.reshape([batch_size, -1, 2])

def data_aug(img1, img2):
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)
    return img1_aug, img2_aug

def build_model(net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motion1,H_motion2, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion1, H_motion2, mesh_motion = net(input1_tensor, input2_tensor)
    H_motion1 = H_motion1.reshape(-1, 4, 2)
    H_motion2 = H_motion2.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion1
    dst_p_total = src_p + H_motion1 + H_motion2
    H1 = torch_DLT.tensor_DLT(src_p, dst_p)
    H_total = torch_DLT.tensor_DLT(src_p,dst_p_total)
    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat1 = torch.matmul(torch.matmul(M_tile_inv, H1), M_tile)
    H_mat_total = torch.matmul(torch.matmul(M_tile_inv,H_total),M_tile)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H1 = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat1, (img_h, img_w))
    output_H_total = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat_total, (img_h, img_w))
    H_inv_mat1 = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H1)), M_tile)
    H_inv_mat_total = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H_total)), M_tile)
    output_H_inv1 = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat1, (img_h, img_w))
    output_H_inv_total = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat_total, (img_h, img_w))
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H_total, rigid_mesh)
    mesh = ini_mesh + mesh_motion
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]
    overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)
    out_dict = {}
    out_dict.update(output_H1=output_H1,
                    output_H_inv1 = output_H_inv1,
                    output_H_total=output_H_total,
                    output_H_inv_total = output_H_inv_total,
                    warp_mesh = warp_mesh,
                    warp_mesh_mask = warp_mesh_mask,
                    mesh1 = rigid_mesh, mesh2 = mesh,
                    overlap = overlap)
    return out_dict

def build_new_ft_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()
    H_motion1, H_motion2, mesh_motion = net(input1_tensor, input2_tensor)
    H_motion1 = H_motion1.reshape(-1, 4, 2)
    H_motion2 = H_motion2.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p1 = src_p + H_motion1
    dst_p2 = dst_p1 + H_motion2
    H_total = torch_DLT.tensor_DLT(src_p, dst_p2)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H_total, rigid_mesh)
    mesh = ini_mesh + mesh_motion
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]
    out_dict = {}
    out_dict.update(warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh)
    return out_dict

def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()
    rigid_mesh = torch.stack([rigid_mesh[..., 0] * img_w / 512, rigid_mesh[..., 1] * img_h / 512], 3)
    mesh = torch.stack([mesh[..., 0] * img_w / 512, mesh[..., 1] * img_h / 512], 3)
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)
    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)
    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:, :, int(torch.abs(height_min)):int(torch.abs(height_min)) + img_h,
    int(torch.abs(width_min)):int(torch.abs(width_min)) + img_w] = (input1_tensor + 1) * 127.5
    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:, :, int(torch.abs(height_min)):int(torch.abs(height_min)) + img_h,
    int(torch.abs(width_min)):int(torch.abs(width_min)) + img_w] = 255
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor + 1, mask], 1), norm_mesh,
                                                     norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:, 0:3, :, :] * 127.5
    mask2 = stitch_tps_out[:, 3:6, :, :] * 255
    stitched = warp1 * (warp1 / (warp1 + warp2 + 1e-6)) + warp2 * (
                warp2 / (warp1 + warp2 + 1e-6))
    stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1, 2, 0),
                                      mesh_trans[0].cpu().detach().numpy())
    out_dict = {}
    out_dict.update(warp1=warp1, mask1=mask1, warp2=warp2, mask2=mask2, stitched=stitched,
                    stitched_mesh=stitched_mesh)
    return out_dict

def build_output_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()
    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion1, H_motion2, mesh_motion = net(resized_input1, resized_input2)
    H_motion1 = H_motion1.reshape(-1, 4, 2)
    H_motion2 = H_motion2.reshape(-1, 4, 2)
    H_motion1 = torch.stack([H_motion1[..., 0] * img_w / 512, H_motion1[..., 1] * img_h / 512], 2)
    H_motion2 = torch.stack([H_motion2[..., 0] * img_w / 512, H_motion2[..., 1] * img_h / 512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)
    mesh_motion = torch.stack([mesh_motion[..., 0] * img_w / 512, mesh_motion[..., 1] * img_h / 512], 3)
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p2 = src_p + H_motion1 + H_motion2
    H_total = torch_DLT.tensor_DLT(src_p, dst_p2)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H_total, rigid_mesh)
    mesh = ini_mesh + mesh_motion
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)
    out_width = width_max - width_min
    out_height = height_max - height_min
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                             [0., out_height / 2.0, out_height / 2.0],
                             [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                             [0., img_h / 2.0, img_h / 2.0],
                             [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)
    I_ = torch.tensor([[1., 0., width_min],
                       [0., 1., height_min],
                       [0., 0., 1.]])
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)
    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor + 1, mask), 1), I_mat,
                                                   (out_height.int(), out_width.int()))
    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor + 1, mask], 1), norm_mesh, norm_rigid_mesh,
                                                 (out_height.int(), out_width.int()))
    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...] - 1, final_warp1_mask=homo_output[:, 3:6, ...],
                    final_warp2=tps_output[:, 0:3, ...] - 1, final_warp2_mask=tps_output[:, 3:6, ...], mesh1=rigid_mesh,
                    mesh2=mesh_trans)
    return out_dict

class HEstimator(nn.Module):
    def __init__(self):
        super(HEstimator, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1024,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=8,bias =True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256,512,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=8,bias=True)
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256,512,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2,bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        ssl._create_default_https_context = ssl._create_unverified_context
        self.feature_extractor = DAFENet()
    def forward(self, input1, input2):
        device = input1.device
        batch_size, _, img_h, img_w = input1.size()
        feature1 = self.feature_extractor(input1)
        feature2 = self.feature_extractor(input2)
        corr_1 = self.CCL(feature1[-1], feature2[-1])
        net1_f = self.net1(corr_1).reshape(batch_size, 4, 2)
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]], device=device)
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + net1_f
        H1 = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
        M_32 = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                          [0., img_h/8 / 2.0, img_h/8 / 2.0],
                          [0., 0., 1.]], device=device)
        M_inv_32 = torch.inverse(M_32)
        M_tile_32 = M_32.unsqueeze(0).expand(batch_size, -1, -1)
        M_tile_inv_32 = M_inv_32.unsqueeze(0).expand(batch_size, -1, -1)
        H1_mat = torch.bmm(torch.bmm(M_tile_inv_32, H1), M_tile_32)
        feature2_warp = torch_homo_transform.transformer(feature2[-3], H1_mat,
                                                        (int(img_h/8), int(img_w/8)))
        corr_2 = self.CCL(feature1[-3], feature2_warp)
        net2_f = self.net2(corr_2).reshape(batch_size, 4, 2)
        dst_p_2 = dst_p + net2_f
        H2 = torch_DLT.tensor_DLT(src_p/8, dst_p_2/8)
        M_16 = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                          [0., img_h/8 / 2.0, img_h/8 / 2.0],
                          [0., 0., 1.]], device=device)
        M_inv_16 = torch.inverse(M_16)
        M_tile_16 = M_16.unsqueeze(0).expand(batch_size, -1, -1)
        M_tile_inv_16 = M_inv_16.unsqueeze(0).expand(batch_size, -1, -1)
        H2_mat = torch.bmm(torch.bmm(M_tile_inv_16, H2), M_tile_16)
        feature3_warp = torch_homo_transform.transformer(feature2[-2], H2_mat,
                                                         (int(img_h/8), int(img_w/8)))
        corr_3 = self.CCL(feature1[-2], feature3_warp)
        net3_f = self.net3(corr_3)
        return net1_f, net2_f, net3_f
    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches
    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()
        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()
        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))
        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)
        match_vol = torch.cat(match_vol, 0)
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)
        channel = match_vol.size()[1]
        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)
        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)
        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)
        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)
        feature_flow = torch.cat([flow_w, flow_h], 1)
        return feature_flow