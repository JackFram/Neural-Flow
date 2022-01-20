from .base import BaseOp
from netflow import *

import numpy as np
import torch
from torch import nn
import copy
class SVDDecomposedLayer():
    def __init__(self, layer,
                 rank = None,
                 pretrained = None):
        self.layer_name = "SVDLinear"
        self.layer = layer
        self.pretrained = pretrained
        
        self.min_rank = 8
       
        if  isinstance(self.layer, nn.Sequential):
            self.in_features = self.layer[0].in_features
            self.out_features = self.layer[1].out_features
        else:
            if not isinstance(self.layer, nn.Linear):
                raise AttributeError('only linear layer can be decomposed')
            self.in_features = self.layer.in_features
            self.out_features = self.layer.out_features
        
        self.weight, self.bias = self.get_weights_to_decompose()
        self.rank = rank
        #### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        # self.new_layers = nn.Linear(in_features = self.in_features, out_features = self.out_features)

        [w0, w1], [_, b] = self.get_svd_factors()        

        # self.new_layers.weight.data = torch.matmul(w1, w0)

        self.recon_w = torch.matmul(w1, w0)

        if b is not None:
            self.recon_b = b
        else:
            self.recon_b = None

        weights, biases = [w0, w1], [_, b]
        for j, (w, b)  in enumerate(zip(weights, biases)):
            self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).weight.data = w
            if b is not None:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias.data = b
            else:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias = None
                
        self.layer = None
        self.weight = None
        self.bias = None
    
    def create_new_layers(self):                       

        layers = [] 
        layers.append(nn.Linear(in_features = self.in_features, 
                                out_features = self.rank,
                                 bias = False))
        layers.append(nn.Linear(in_features = self.rank, 
                                out_features = self.out_features))
        return layers
    
    def get_weights_to_decompose(self):
        if  isinstance(self.layer, nn.Sequential):
            #weight = self.layer[1].weight.data @ self.layer[0].weight.data 
            weight = self.layer[1].weight.data
            try:
                bias = self.layer[1].bias.data
            except:
                bias = None
        else:
            weight = self.layer.weight.data
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight, bias
    
        
    def get_svd_factors(self):
        if self.pretrained is not None:
            raise AttributeError('Not implemented')
        else:
            weights = self.weight.cpu()
            if self.bias is not None:
                bias = self.bias.cpu()
            else:
                bias = self.bias
                
            U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)
            
            w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
            w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))
            
            if  isinstance(self.layer, nn.Sequential):
                w0_old = self.layer[0].weight.cpu().data
                w0 = np.dot(w0, w0_old)
                
            w0 = torch.FloatTensor(w0).contiguous()
            w1 = torch.FloatTensor(w1).contiguous()
            

        return [w0, w1], [None, bias]

class SVDDecomposedConvLayer():
    def __init__(self, layer, layer_name,
                 rank = None,
                 pretrained = None):

        self.layer_name = "SVDConv"
        self.layer = layer
        self.pretrained = pretrained
        
        self.min_rank = 2
        
        #print(layer)
       
        if  isinstance(self.layer, nn.Sequential):
            self.in_channels = self.layer[0].in_channels
            self.out_channels = self.layer[1].out_channels
            
            self.padding = self.layer[1].padding
            self.stride = self.layer[1].stride
        else:
            if not isinstance(self.layer, nn.Conv2d):
                raise AttributeError('only conv layer can be decomposed')
            self.in_channels = self.layer.in_channels
            self.out_channels = self.layer.out_channels
            
            self.padding = self.layer.padding
            self.stride = self.layer.stride
        
        self.weight, self.bias = self.get_weights_to_decompose()
        self.rank = rank
            
        #### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        # self.new_layers = nn.Conv2d(in_channels = self.in_channels,
        #                             out_channels = self.out_channels,
        #                             padding = self.layer.padding,
        #                             stride = self.layer.stride)
        [w0, w1], [_, b] = self.get_svd_factors()  

        self.recon_w = torch.matmul(w1, w0)

        if b is not None:
            self.recon_b = b
        else:
            self.recon_b = None

        weights, biases = [w0, w1], [_, b]
        for j, (w, b) in enumerate(zip(weights, biases)):
            self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).weight.data = w
            if b is not None:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias.data = b
            else:
                self.new_layers.__getattr__('{}-{}'.format(self.layer_name, j)).bias = None
                
        self.layer = None
        self.weight = None
        self.bias = None
    
    def create_new_layers(self):                       

        layers = [] 
        layers.append(nn.Conv2d(in_channels = self.in_channels, 
                                out_channels = self.rank,
                                kernel_size = 1, 
                                bias = False))
        layers.append(nn.Conv2d(in_channels = self.rank,  
                                out_channels = self.out_channels,
                                kernel_size = 1,
                                padding = self.padding,
                                stride = self.stride))
        return layers
    
    def get_weights_to_decompose(self):
        if  isinstance(self.layer, nn.Sequential):
            #weight = self.layer[1].weight.data @ self.layer[0].weight.data 
            weight = self.layer[1].weight.data
            try:
                bias = self.layer[1].bias.data
            except:
                bias = None
        else:
            weight = self.layer.weight.data
            try:
                bias = self.layer.bias.data
            except:
                bias = None
        return weight[:,:,0,0], bias
    
        
    def get_svd_factors(self):
        if self.pretrained is not None:
            raise AttributeError('Not implemented')
        else:
            weights = self.weight.cpu()
            if self.bias is not None:
                bias = self.bias.cpu()
            else:
                bias = self.bias
                
            U, S, Vt = np.linalg.svd(weights.data.numpy(), full_matrices=False)
            
            w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
            w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))
            
            if  isinstance(self.layer, nn.Sequential):
                w0_old = self.layer[0].weight[:,:,0,0].cpu().data
                w0 = np.dot(w0, w0_old)
                
            w0 = torch.FloatTensor(w0[:,:, np.newaxis, np.newaxis]).contiguous()
            w1 = torch.FloatTensor(w1[:,:, np.newaxis, np.newaxis]).contiguous()
            

        return [w0, w1], [None, bias]

class LowRankOp(BaseOp):
    def __init__(self, model: nn.Module, rank_fraction=1):
        super().__init__(model)
        self.op_name = "lowrank"
        self.rank_fraction = rank_fraction
        self.mod_model = None

    def apply(self, name_list, rank_fraction=None, verbose=False, with_profile=False, *args, **kwargs):
        if rank_fraction is None:
           rank_fraction = self.rank_fraction 
        diff = {}
        storage_save = {}
        model_to_lowrank = copy.deepcopy(self.model)
        for name in set(name_list):
            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError
            
            prevmod_name = ".".join(name.split(".")[:-1])
            mod_name = name.split(".")[-1]
            prevmod = model_to_lowrank.get_submodule(prevmod_name)
            mod = model_to_lowrank.get_submodule(name)

            if with_profile:
                param = self.get_param(mod)

            dim_in, dim_out = mod.weight.data.cpu().numpy().shape
            rank = int(rank_fraction * dim_in * dim_out / (dim_in + dim_out))

            if rank_fraction < 1:
                svd_mod = self.low_rank_(prevmod, mod_name, mod, rank)
                if with_profile:
                    param_ = self.get_param_(svd_mod)
                # print(prevmod, param.shape, param_.shape)
            elif with_profile:
                param_ = param

            if with_profile:
                diff[name] = param - param_
                # print(rank_fraction, rank, dim_in, dim_out, np.linalg.norm(diff[name]), np.abs(diff[name]).max())
                storage_save[name] = rank_fraction

        self.mod_model = model_to_lowrank
        if with_profile:
            return self.mod_model, diff, storage_save
        else:
            return self.mod_model

    def low_rank_(self, prev_layer, layer_name, layer, rank):
        if isinstance(layer, nn.Conv2d):
            decomposed_layer = SVDDecomposedConvLayer(layer, layer_name, rank)
        elif isinstance(layer, nn.Linear):
            decomposed_layer = SVDDecomposedLayer(layer, rank)
        prev_layer.__setattr__(layer_name, decomposed_layer.new_layers)
        return decomposed_layer

    def get_param(self, mod:nn.modules):
        weight = mod.weight.data.cpu().numpy().flatten()
        if hasattr(mod, "bias") and mod.bias is not None:
            bias = mod.bias.data.cpu().numpy().flatten()
            return np.concatenate([weight, bias], axis=0)
        return weight

    def get_param_(self, svd_mod):
        weight = svd_mod.recon_w.cpu().numpy().flatten()
        if hasattr(svd_mod, "recon_b") and svd_mod.recon_b is not None:
            bias = svd_mod.recon_b.cpu().numpy().flatten()
            return np.concatenate([weight, bias], axis=0)
        return weight

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
