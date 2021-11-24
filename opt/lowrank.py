from .base import BaseOp
from netflow import *

import numpy as np
import torch
from torch import nn
import copy


class SVDDecomposedLayer():
    def __init__(self, layer, layer_name,
                 rank = None,
                 pretrained = None):

        self.layer_name = layer_name
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
            
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_svd_factors()        
        
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

        self.layer_name = layer_name
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
            
        ##### create decomposed layers
        self.new_layers = nn.Sequential()
        
        for j, l in enumerate(self.create_new_layers()):
            self.new_layers.add_module('{}-{}'.format(self.layer_name, j), l)
        
        weights, biases = self.get_svd_factors()        
        
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


def get_compressed_model(model,
                         layer_names,
                         rank):
    '''
    layer_names:list,
    ranks: defaultdict,
    decompositions: defaultdict,
    layer_types: defaultdict,
    vbmf_weaken_factors: defaultdict
    '''
    compressed_model = copy.deepcopy(model)
    model = None

    for mod_name, mod in compressed_model.named_modules():

        if mod_name in layer_names:

            ## model before 
            #print('subm_name: {} \n'.format(subm_names))
            layer = compressed_model.__getattr__(mod_name)

            if isinstance(mod, nn.Conv2d):
                decomposed_layer = SVDDecomposedConvLayer(layer,\
                                                            layer,\
                                                            rank)
            elif isinstance(mod, nn.Linear):
                decomposed_layer = SVDDecomposedLayer(layer,\
                                                        layer,\
                                                        rank)

            
            compressed_model.__setattr__(layer, decomposed_layer.new_layers)

    return compressed_model

class LowRankOp(BaseOp):
    def __init__(self, model: nn.Module, rank=10):
        super().__init__(model)
        self.op_name = "low rank"
        self.rank = rank
        self.mod_model = None

    def apply(self, name_list, verbose=False, *args, **kwargs):
        name_set = set()
        for name in name_list:
            if name not in self.operatable:
                print("{} is not a operatable layer, retry something in:{} !".format(name, self.operatable))
                raise AttributeError
            name_set.add(name)

        self.mod_model = get_compressed_model(self.model, name_set, self.rank)

        return self.mod_model

    @property
    def operatable(self):
        ret_list = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
                ret_list.append(name)
        return ret_list
