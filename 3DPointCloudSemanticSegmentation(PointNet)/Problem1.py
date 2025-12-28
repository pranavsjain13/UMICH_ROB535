"""
Code written by Joey Wilson, 2023.
"""

from sys import _xoptions
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset


class PointLoader(Dataset):
  def __init__(self, root, remap_array, data_split="Train",
              device="cpu", max_size=40000):
    self.root = root
    self.velo_dir = os.path.join(root, "velodyne_ds")
    self.velo_files = sorted(os.listdir(self.velo_dir))
    self.split = data_split
    if not self.split == "Test":
      self.label_dir = os.path.join(root, "labels_ds")
      self.label_files = sorted(os.listdir(self.label_dir))
    self.device = device
    self.remap_array = remap_array
    self.max_size = max_size
  
  def __len__(self):
    return len(self.velo_files)

  def __getitem__(self, index):
    # Load point cloud
    velo_file = os.path.join(self.velo_dir, self.velo_files[index])
    # TODO: Fetch the point cloud
    pc = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
    # Test set has no available ground truth
    if self.split == "Test":
      return pc, None
    # Load labels
    label_file = os.path.join(self.label_dir, self.label_files[index])
    # TODO: Fetch labels from label_file
    label = np.fromfile(label_file, dtype=np.uint32)
    # TODO: Mask the label with a mask of 0xFFFF
    masked_label = label & 0xFFFF
    # TODO: Use the masked label to index the remap array
    label = self.remap_array[masked_label]
    # Downsample the points
    if self.split == "Train":
      indices = np.random.permutation(pc.shape[0])[:self.max_size]
      pc = pc[indices, :]
      label = label[indices]
    return pc, label

  def collate_fn(self, data):
    B = len(data)
    pc_numpy = [data[i][0] for i in range(B)]
    torch_pc = torch.tensor(np.array(pc_numpy), device=self.device)
    label_numpy = [data[i][1] for i in range(B)]
    torch_label = torch.tensor(np.array(label_numpy), device=self.device)
    return torch_pc, torch_label 


class PointNetEncoder(nn.Module):
  def __init__(self, cs, linear_out=None):
    super().__init__()
    # cs is a list of channel sizes e.g. [3, 64, num_classes]
    # Each layer i contains a linear layer from 
    # cs [i-1] to cs[i]
    # and a ReLU nonlinearity 
    # linear_out is in the case this is the last layer of the network
    self.net = torch.nn.Sequential()
    for i in range(1, len(cs)):
      # TODO: Replace None with a linear layer from cs[i-1] to cs[i]
      self.net.add_module("Lin" + str(i), nn.Linear(cs[i-1], cs[i]))
      # TODO: Replace None with a batchnorm layer of size cs[i]
      self.net.add_module("Bn" + str(i), nn.BatchNorm1d(cs[i]))
       #TODO: Replace None with a ReLU Layer
      self.net.add_module("ReLU" + str(i), nn.ReLU())
    if linear_out is not None:
      # TODO: Replace None with a linear layer from cs[i] to linear_out
      self.net.add_module("LinFinal", nn.Linear(cs[i], linear_out))

  def forward(self, x):
    # Input x is a BxNxC matrix where N is number of points
    B, N, C = x.shape
    # TODO: Use x.view() to reshape x into (B*N, C)
    x = x.view(B*N, C)
    # TODO: Feed x through your network
    x = self.net(x)
    # TODO: Reshape x into shape (B, N, C1) where C1 is the new channel dim
    x = x.view(B, N, -1)
    return x
    
# This module learns to combines global and local point features
class PointNetModule(nn.Module):
  def __init__(self, cs_en, cs_dec, num_classes=20):
    super().__init__()
    # TODO: Create a PointNetEncoder with cs=cs_en
    self.enc =  PointNetEncoder(cs=cs_en)
    # TODO: Create a PointNetEncoder decoder module with cs=cs_dec
    # and linear_out=num_classes
    self.dec = PointNetEncoder(cs=cs_dec, linear_out=num_classes)

  def forward(self, x):
    B, N, C = x.shape
    # Encoder
    # TODO: Feed x through the PointNetEncoder
    x = x[:, :, :3]
    point_feats = self.enc(x)
    # Max across points
    # TODO: Use max pooling across the point dimension to create
    # a global_feats tensor of shape (B, C1)
    # We used the torch.max
    global_feats, _ = torch.max(point_feats, dim=1)
    # TODO: Reshape global_feats from (B, C1) to (B, 1, C1)
    # And repeat along the middle dimension N times using repeat
    # End tensor should be (B, N, C1)
    global_feats = global_feats.unsqueeze(1).repeat(1, N, 1)
    # TODO: Concatenate local and global features along channel dimension (2)
    joint_feats = torch.cat([point_feats, global_feats], dim=2)
    # TODO: Feed joint_feats through the decoder module
    out = self.dec(joint_feats)
    return out
    
# This module adds a T-Net
class PointNetFull(nn.Module):
  def __init__(self, cs_en, cs_dec, cs_t_en, cs_t_dec):
    super().__init__()
    # TODO: Create a PointNetEncoder with cs=cs_t_en
    self.t_enc = PointNetEncoder(cs=cs_t_en)
    # TODO: Create a PointNetEncoder with cs=cs_t_dec and linear_out = 9
    # Note that 9 comes from the product 3x3
    self.t_dec = PointNetEncoder(cs=cs_t_dec, linear_out=9)
    # TODO: Create a PointNetModule with cs_en, and cs_dec
    self.joint_enc = PointNetModule(cs_en, cs_dec)

  def forward(self, x):
    B, N, C = x.shape
    # T-Net
    # TODO: Feed x through the t-net encoder
    x = x[:, :, :3]
    t_feats = self.t_enc(x)
    # TODO: Max pool across the point dimension of t_feats
    t_feats, _ = torch.max(t_feats, dim=1)
    # TODO: Reshape t_feats to shape (B, 1, C1)
    t_feats = t_feats.unsqueeze(1)
    # TODO: Feed t_feats through the T-Net Decoder
    t_feats = self.t_dec(t_feats)
    # TODO: Reshape t_feats to (B, 3, 3)
    point_trans = t_feats.view(B, 3, 3)
    # TODO: Compute the transformation (B, 3, 3) matrices
    # As a summation of point_trans and an identity matrix
    point_trans = point_trans + torch.eye(3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    # Apply transform
    # TODO: Perform batched matrix multiplication between x and point_trans
    # torch.bmm() may be helpful
    transformed_points = torch.bmm(x, point_trans)
    # Joint Encoder
    # TODO: Feed the transformed_points through the joint encoder
    output_preds = self.joint_enc(transformed_points)
    return output_preds


# Compute the per-class iou and miou
def IoU(targets, predictions, num_classes, ignore_index=0):
  intersections = torch.zeros(num_classes, device=targets.device)
  unions = torch.zeros_like(intersections)
  counts = torch.zeros_like(intersections)
  # TODO: Discard ignored points
  valid_mask = (targets != ignore_index)
  targets = targets[valid_mask]
  predictions = predictions[valid_mask]
  # Loop over classes and update the counts, unions, and intersections
  for c in range(num_classes):
    # TODO: Fill in computation
    target_mask = (targets == c)
    pred_mask = (predictions == c)
    intersections[c] = torch.sum(target_mask & pred_mask)
    unions[c] = torch.sum(target_mask | pred_mask)
    # Add small value to avoid division by 0
    # Make sure to keep the small smoothing constant to match the autograder
    unions[c] = unions[c] + 1e-8
  # Per-class IoU
  iou = intersections / unions
  # Make sure to set iou for classes with no points to 1
  iou[unions == 1e-8] = 1.0
  # Calculate mean, ignoring ignore index
  valid_classes = (unions > 1e-8) & (torch.arange(num_classes, device=targets.device) != ignore_index)
  miou = torch.mean(iou[valid_classes]) if torch.any(valid_classes) else torch.tensor(0.0)
  return iou, miou

    

