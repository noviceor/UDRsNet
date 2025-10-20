import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter
import os
class IACNet(nn.Module):
    def __init__(self):
        super().__init__()   
        # branch1: 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GroupNorm(16, 64),  # GroupNorm
            nn.Conv2d(64, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 192),  # GroupNorm
            nn.ReLU(inplace=True)
        )
        
        # branch2: 5x5 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.Conv2d(64, 96, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
        )
        
        # branch3: maxpool + 1x1 convolution
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 64, 1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ], dim=1)

class DAFENet(nn.Module):
    def __init__(self,  out_channels=256,pretrained_path='../../resnet50-11ad3fa6.pth'):
        """
        Build a feature extractor using pretrained ResNet50 and FPN, returning feature maps as a list.

        Args:
            train_backbone (bool): Whether to train backbone parameters. Default is False.
            out_channels (int): Number of output channels for FPN. Default is 256.
        """
        super(DAFENet, self).__init__()
        #pretrained_path = './data/coding/resnet50-11ad3fa6.pth'
        # Load pretrained ResNet50
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from local file: {pretrained_path}")
            # Create model without pretrained weights
            backbone = resnet50(weights=None)
            # Load local pretrained weights
            #state_dict = torch.load(pretrained_path,weights_only=True)
            state_dict = torch.load(pretrained_path)
            backbone.load_state_dict(state_dict)
        else:
            print("Local pretrained weights not found. Loading from torchvision...")
            try:
                # Use pretrained weights from torchvision
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except AttributeError:
                # For older torchvision versions, use legacy loading
                backbone = resnet50(pretrained=True)

        # Define intermediate layers to extract
        # ResNet50's layer1, layer2, layer3, layer4 output features at different scales
        returned_layers = { 'layer1': '0','layer2': '1', 'layer3': '2'}
        self.body = IntermediateLayerGetter(backbone, return_layers=returned_layers)
        # Output channels for each ResNet50 layer
        in_channels_list = [256,512, 1024]

        # Build FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=None  # Add extra blocks if needed
        )
        self.downsample_p2 = IACNet()
    def forward(self, x):
        """
        Forward pass, returns multi-scale features from FPN.
        Args:
            x (Tensor): Input image, shape (B, 3, H, W)
        Returns:
            List[Tensor]: Multi-scale feature maps [P2, P3, P4, P5]
            P2: (128, 128, 256) 1/4
            P3: (64, 64, 256) 1/8
            P4: (32, 32, 256) 1/16
            P5: (16, 16, 256) 1/32
        """
        # Get intermediate features from ResNet
        features = self.body(x)
        # Build feature pyramid using FPN
        fpn_features = self.fpn(features)
        # Sort and return feature maps in order
        # Assume returned_layers order is ['0', '1', '2', '3']
        # Corresponding feature map order: [P2, P3, P4, P5]
        p2 = fpn_features['0']  # P2: (B, 256, 128, 128)
        p2_downsampled = self.downsample_p2(p2)  # P2: (B, 256, 64, 64)
        # Return adjusted feature maps
        out_features = [p2_downsampled, fpn_features['1'], fpn_features['2']]
        return out_features


