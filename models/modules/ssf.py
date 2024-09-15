#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
# Adapted from LoRA layers

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSFLayer():
    def __init__(
        self, 
        merge_weights: bool,
    ):

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class SSFLinear(nn.Linear, SSFLayer):
    # SSF implemented in a dense layer
    def __init__(
        self, 
        layer: nn.Linear, 
        merge_weights: bool = True,
    ):
        out_features, in_features = layer.weight.shape
        nn.Linear.__init__(self, in_features, out_features)
        SSFLayer.__init__(self, merge_weights=merge_weights)

        # Actual trainable parameters
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

        self.weight = layer.weight
        self.bias = layer.bias
        
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'scale'):
            nn.init.normal_(self.scale, mean=1, std=0.02)
            nn.init.normal_(self.shift, std=.02)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data = self.weight.data / self.scale.view(-1, 1)
                self.bias.data = (self.bias.data - self.shift) / self.scale

                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data = self.weight.data * self.scale.view(-1, 1)
                self.bias.data = self.bias.data * self.scale + self.shift

                self.merged = True       

    def forward(self, x: torch.Tensor):
        if not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)            
            result = result * self.scale + self.shift
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)


class SSFConv1d(nn.Conv1d, SSFLayer):
    def __init__(self, layer: nn.Conv1d, merge_weights=True, **kwargs):
        out_channels, in_channels, kernel_size = layer.weight.shape
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=layer.stride,padding=layer.padding, dilation=layer.dilation, groups=layer.groups, bias=False if layer.bias is None else True**kwargs)
        SSFLayer.__init__(self, merge_weights=merge_weights)

        # Actual trainable parameters
        self.scale = nn.Parameter(torch.ones(out_channels))
        if layer.bias:
            self.shift = nn.Parameter(torch.zeros(out_channels))
            
        self.reset_parameters()

        self.weight = layer.weight
        self.bias = layer.bias

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        if self.bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        nn.Conv1d.reset_parameters(self)
        if hasattr(self, 'scale'):
            nn.init.normal_(self.scale, mean=1, std=0.02)
            if self.bias:
                nn.init.normal_(self.shift, std=.02)

    def train(self, mode=True):
        nn.Conv1d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data = self.weight.data / self.scale.view(-1, 1, 1)
                if self.bias:
                    self.bias.data = (self.bias.data - self.shift.view(-1, 1, 1)) / self.scale.view(-1, 1, 1)

                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data = self.weight.data * self.scale.view(-1, 1, 1)
                if self.bias:
                    self.bias.data = self.bias.data * self.scale.view(-1, 1, 1) + self.shift.view(-1, 1, 1)

                self.merged = True  

    def forward(self, x):
        if not self.merged:
            result = F.conv1d(x, self.weight, self.bias)
            result = result * self.scale.view(1, -1, 1) + self.shift if self.bias else result * self.scale.view(1, -1, 1)
            return result
        return F.conv1d(x, self.weight, self.bias)
