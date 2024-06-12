import numpy as np
import torch
from sklearn.model_selection import train_test_split
from numpy import load
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader,Dataset,ConcatDataset,Sampler
import matplotlib.pyplot as plt
from numpy import load
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.init as init
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import copy
import scikit_posthocs as sp
from einops import rearrange
import json
import os
import sys
import builtins
import functools
import time
import random
from copy import deepcopy
import math
import pandas as pd
from matplotlib.colors import Normalize
import pickle
from itertools import combinations
from itertools import product

import torch.nn.functional as F
from torch.autograd import Function





import time
from sklearn.metrics import f1_score, confusion_matrix

L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)


#utilitaires 
rep_geo={f'{R2018}':'R18',f'{R2019}':'R19',f'{R2020}':'R20',f'{T2018}':'T18',f'{T2019}':'T19',f'{T2020}':'T20',f'{L2018}':'L18',f'{L2019}':'L19',f'{L2020}':'L20'}

class customdata(Dataset):
  def __init__(self,values,labels):
    self.values=values
    self.labels=labels
  def __len__(self):
    return len(self.values)
  def  shape(self):
    return self.values.shape
  def __getitem__(self,id):
    value=self.values[id]
    label=self.labels[id]
    return value, label


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif -val_loss > -(self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print("Saved new best model.")
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False



def get_day_count(dates,ref_day='09-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int) #(dates - ref_day).astype('timedelta64[D]').astype(int)#
    return torch.tensor(days_elapsed,dtype=torch.long)

def add_mask(values,mask): # permet d'attacher les mask aux données pour pouvoir faire les batchs sans perdre le mask
    mask=mask.unsqueeze(0).unsqueeze(-1)
    shape=values.shape
    mask=mask.expand(shape[0],-1,-1)
    values=torch.tensor(values,dtype=torch.float32)

    valuesWmask=torch.cat((values,mask),dim=-1)
    return valuesWmask

def comp (data,msk) : #permet de formater les données avec 365 points d'acquisitions
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  j_p=(data['dates_SAR']-ref).astype('timedelta64[D]').astype(int)
  année=list(range(365))

  année = [ref + np.timedelta64(j, 'D') for j in année ]
  mask = []

  for i,jour in enumerate(année):
    if jour not in data['dates_SAR']:

      mask+=[0]
      msk=np.insert(msk,i,0)
      data_r['dates_SAR']=np.insert(data_r['dates_SAR'],i,jour)
      data_r['X_SAR']=np.insert(data_r['X_SAR'],i,[0,0],axis=1)
    else:
      mask+=[1]


  mask=torch.tensor(mask,dtype=torch.float32)
  msk=torch.tensor(msk,dtype=torch.float32)
  return data_r,mask,msk


def suppr (data,ratio):
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  nbr,seq_len,channels=data['X_SAR'].shape #(nbr,seq_len,channels)
  print(f'ratio ={ratio}')
  nbr_indice=int(seq_len*ratio)
  indice=list(range(seq_len))
  indice=random.sample(indice,nbr_indice)
  mask=[0 if i in indice else 1 for i in range(seq_len)]
  mask=torch.tensor(mask)

  data_r['X_SAR']=torch.tensor(data_r['X_SAR'])
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].masked_fill(mask==0,0)
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].numpy()
  mask=mask.numpy()
  return data_r,mask


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0,dates=None):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dates=dates

        pe = torch.zeros(max_len, d_model)  # positional encoding

        if dates is None :
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        else:
          position=get_day_count(dates).unsqueeze(1)
        print(position.shape,pe.shape,max_len)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        if mask is not None :
            s=attn.shape

            mask=mask.unsqueeze(1)
            mask=mask.repeat(1,s[1],1)
            mask=mask.unsqueeze(2)
            mask=mask.repeat(1,1,s[2],1)
            attn=attn.masked_fill(mask==0,-10000)

        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model



class CasualConvTran(nn.Module):
    def __init__(self, config, num_classes,dates):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.dates=dates
        dropout=config['dropout']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.Conv1 = nn.Sequential(nn.Conv1d(channel_size, emb_size, kernel_size=3,padding=1, stride=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.Conv2 = nn.Sequential(nn.Conv1d(emb_size, emb_size, kernel_size=3, stride=1,padding=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(nn.Conv1d(emb_size, emb_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout, seq_len,dates=dates)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode =='tAPE_sansDC':
          self.Fix_Position = tAPE_sansDC(emb_size,dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x,mask):
        #x = x.unsqueeze(1)
        x=torch.permute(x,(0,2,1))
        x_src = self.Conv1(x)
        x_src = self.Conv2(x_src)#.squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos,mask)
        else:
            att = x_src + self.attention_layer(x_src,mask)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out1= self.flatten(out)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out,out1

"""# REFeD"""

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
    #def __init__(self, temperature=1., min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.t_period = t_period
        self.eps = eps

    def forward(self, projections, targets, epoch=1):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")


        dot_product = torch.mm(projections, projections.T)

        dot_product_tempered = dot_product / self.temperature

        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        stab_max, _ = torch.max(dot_product_tempered, dim=1, keepdim=True)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - stab_max.detach() ) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        #### FILTER OUT POSSIBLE NaN PROBLEMS ####
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### ####

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss


class ConvTranRD(torch.nn.Module):
    def __init__(self,config, num_classes=11,num_dom=8,dates=None):
        super(ConvTranRD, self).__init__()

        self.inv = CasualConvTran(config,num_classes=num_classes,dates=dates)
        self.spec = CasualConvTran(config,num_classes=num_dom,dates=dates)
        self.dom_rev = nn.LazyLinear(num_dom)

    def forward(self, x,mask,alpha=1):
        classif, inv_emb = self.inv(x,mask)
        reverse_emb = ReverseLayerF.apply(inv_emb, alpha)
        adv_cl_dom = self.dom_rev(reverse_emb)
        classif_spec, spec_emb = self.spec(x,mask)
        return classif, inv_emb, spec_emb, classif_spec, adv_cl_dom

# @title préparation data
def prep_data_(data,ratio=0,ratio_supprimé=0): # datas doit être une liste contenant 1 ou 2 jeux de donnés

    mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10}


    data,msk=suppr(data,ratio_supprimé) # peut être utiliser si l'on souhaite diminuer la quantité de points d'acquisition dans les données
    data,_,mask=comp(data,msk) # rempli les données pour les mettre au fromat 365 j et donne le mask correspondant aux jours où on a mit un 0
    values=data['X_SAR']
    data_shape=data['X_SAR'].shape
    dates=data['dates_SAR']




    labels=data['y']
    labels=[mapping[v] if v in mapping else v for v in labels ]

    max_values = np.percentile(values,99)
    min_values = np.percentile(values,1)
    values_norm=(values-min_values)/(max_values-min_values)
    values_norm[values_norm>1] = 1
    values_norm[values_norm<0] = 0
    values = values_norm                                      # les données sont normalisées
    values=add_mask(values,mask)                              # ajoute le mask aux données pour qu'il soit disponiuble dans le dataloader

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indice = sss.split(values,labels)

    tv_index, test_index = next(indice)

    values_tv=[]
    values_test=[]
    labels_tv=[]
    labels_test=[]
    for i in tv_index :
        values_tv+=[values[i]]
        labels_tv+=[labels[i]]
    for j in test_index :
        values_test+=[values[j]]
        labels_test+=[labels[j]]


    sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=0)
    indice2=sss2.split(values_tv,labels_tv)
    train_index,validation_index = next(indice2)

    values_train=[]
    values_validation=[]
    labels_train=[]
    labels_validation=[]

    for i in train_index :
        values_train+=[values_tv[i]]
        labels_train+=[labels_tv[i]]
    for j in validation_index :
        values_validation += [values_tv[j]]
        labels_validation += [labels_tv[j]]


    values_train=np.array(values_train)
    values_validation=np.array(values_validation)
    values_test=np.array(values_test)
    labels_train=np.array(labels_train)
    labels_validation=np.array(labels_validation)
    labels_test=np.array(labels_test)







    return values_train,values_validation,values_test,labels_train,labels_validation,labels_test,dates,data_shape



def data_loading_source(data_source):
        values_train = []
        labels_train = []
        labels_domain_train = []
        
        for i, data in enumerate(data_source):
            values_train_source, _, _, labels_train_source, _, _, dates, data_shape = prep_data_(data)
            
            values_train.append(values_train_source)
            labels_train.append(labels_train_source)
            labels_domain_train.append(np.ones(labels_train_source.shape[0])*i)

        # Convert lists to numpy arrays after concatenation
        values_train = np.concatenate(values_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        labels_domain_train = np.concatenate(labels_domain_train, axis=0)



        x_train=torch.tensor(values_train,dtype=torch.float32)
        y_train=torch.tensor(labels_train,dtype=torch.int64)
        dom_train=torch.tensor(labels_domain_train,dtype=torch.int64)




        train_dataset = TensorDataset(x_train, y_train, dom_train)


        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)


        

        return train_dataloader,data_shape,dates
def data_loading_target(data_target):
      values_train_target,values_validation_target,values_test_target,labels_train_target,labels_validation_target,labels_test_target,dates,data_shape_target=prep_data_(data_target)
      values_test=values_test_target
      labels_test=labels_test_target
      x_test=torch.tensor(values_test,dtype=torch.float32)
      y_test=torch.tensor(labels_test,dtype=torch.int64)
      test_dataset = TensorDataset(x_test, y_test)
      test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)
      target_data=rep_geo[f'{data_target}']
      return test_dataloader,target_data,dates

# @title Workflow

def sim_dist_specifc_loss_spc(spec_emb, ohe_label, ohe_dom, scl, epoch):
    norm_spec_emb = nn.functional.normalize(spec_emb)
    hash_label = {}
    new_combined_label = []
    for v1, v2 in zip(ohe_label, ohe_dom):
        key = "%d_%d"%(v1,v2)
        if key not in hash_label:
            hash_label[key] = len(hash_label)
        new_combined_label.append( hash_label[key] )
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_spec_emb, new_combined_label, epoch=epoch)
def sup_contra_Cplus2_classes(emb, ohe_label, ohe_dom, scl, epoch):
    norm_emb = nn.functional.normalize(emb)
    C = ohe_label.max() + 1
    new_combined_label = [v1 if v2==8 else C+v2 for v1, v2 in zip(ohe_label, ohe_dom)]
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_emb, new_combined_label, epoch=epoch)

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for xm_batch, y_batch in dataloader:
        x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        pred = model(x_batch,mask_batch)[0]
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels



def global_loop(data_source,epochs):
        nom=''
        for data_set in data_source:
            a0=rep_geo[f'{data_set}']
            print(rep_geo[f'{data_set}'])
            nom+=f'+{a0}'
        train_dataloader,data_shape_source,dates=data_loading_source(data_source)
        
        n_classes=11
        dim_ff=64
        data_shape=(data_shape_source[0],data_shape_source[2],data_shape_source[1])
        config={'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':dim_ff}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvTranRD(config,n_classes,dates).to(device)



        learning_rate = 0.0001
        loss_fn = nn.CrossEntropyLoss()
        scl = SupervisedContrastiveLoss()

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


        valid_f1 = 0.
        
        

        for epoch in range(epochs):
            start = time.time()
            model.train()
            tot_loss = 0.0
            domain_loss = 0.0
            contra_tot_loss = 0.0
            den = 0

            for xm_batch, y_batch, dom_batch in train_dataloader:
                if xm_batch.shape[0] != 64:
                    continue

                x_batch,m_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
                x_batch = x_batch.to(device)
                m_batch = m_batch.to(device)
                y_batch = y_batch.to(device)
                dom_batch = dom_batch.to(device)
                optimizer.zero_grad()
                pred, inv_emb, spec_emb_d, spec_d_pred,adv_d_pred = model(x_batch,m_batch)

                

                ##### DOMAIN CLASSIFICATION #####
                loss_ce_spec_dom = loss_fn(spec_d_pred, dom_batch)
                loss_avd_dom = loss_fn(adv_d_pred,dom_batch)

                ##### MIXED MAINFOLD & CONTRASTIVE LEARNING ####

                cl_labels_npy = y_batch.cpu().detach().numpy()
                y_mix_labels = np.concatenate([ cl_labels_npy , cl_labels_npy],axis=0)

                #DOMAIN LABEL FOR DOMAIN-CLASS SPECIFIC EMBEDDING and DOMAIN SPECIFIC EMBEDDING IS 0 OR 1
                spec_dc_dom_labels = dom_batch.cpu().detach().numpy()
                #DOMAIN LABEL FOR INV EMBEDDING IS 8 IF 8 DATASETS 
                inv_dom_labels = np.ones_like(spec_dc_dom_labels) * 8

                dom_mix_labels = np.concatenate([inv_dom_labels, spec_dc_dom_labels],axis=0)
                joint_embedding = torch.concat([inv_emb, spec_emb_d])

                #mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)# k*(d+1) 
                mixdl_loss_supContraLoss = sup_contra_Cplus2_classes(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
                




                contra_loss = mixdl_loss_supContraLoss

                ####################################

                loss = loss_fn(pred, y_batch) + contra_loss + loss_ce_spec_dom + loss_avd_dom

                loss.backward() # backward pass: backpropagate the prediction loss
                optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
                tot_loss+= loss.cpu().detach().numpy()
                contra_tot_loss+= contra_loss.cpu().detach().numpy()
                den+=1.


            end = time.time()
            #pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
            #f1_val = f1_score(labels_valid, pred_valid, average="weighted")
            #if f1_val > valid_f1:
            torch.save(model.state_dict(), f"model_REFeD_8{nom}.pth")
                #valid_f1 = f1_val
                #pred_test, labels_test = evaluation(model, test_dataloader, device)
                #f1 = f1_score(labels_test, pred_test, average="weighted")
                
            print(" at Epoch %d: training time %d"%(epoch+1, (end-start)))
                #print(confusion_matrix(labels_test, pred_test))
            #else:
             #   print("TOT AND CONTRA AND TRAIN LOSS at Epoch %d: %.4f %.4f"%(epoch+1, tot_loss/den, contra_tot_loss/den))
            #sys.stdout.flush()
        return model



def test_loop(modèle,data_target):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      test_dataloader,target_data,_=data_loading_target(data_target)
      pred_test, labels_test = evaluation(modèle, test_dataloader, device)
      f1 = f1_score(labels_test, pred_test, average="weighted")
      return f1




def final_test(listes_données,epochs):
  dict_refed8={}
  rep_geo={f'{R2018}':'R18',f'{R2019}':'R19',f'{R2020}':'R20',f'{T2018}':'T18',f'{T2019}':'T19',f'{T2020}':'T20',f'{L2018}':'L18',f'{L2019}':'L19',f'{L2020}':'L20'}
  jeu_test=list(combinations(listes_données,8))
  #jeu_filtré=[elem for elem in jeu_test if rep_geo[f'{elem[0]}'][0]==rep_geo[f'{elem[1]}'][0]  ]
  #ensemble=[(elem[0],elem[1],test) for elem,test in product(jeu_filtré,listes_données)]# if rep_geo[f'{test}'] not in [rep_geo[f'{elem[0]}'],rep_geo[f'{elem[1]}'] ]]
  #dbg=[(rep_geo[f'{ens[0]}'],rep_geo[f'{ens[1]}'],rep_geo[f'{ens[2]}'])for ens in ensemble if  rep_geo[f'{ens[2]}'] not in [rep_geo[f'{ens[0]}'],rep_geo[f'{ens[1]}']] ]
  #for dd in dbg:
  #    print(dd)
  #return (dbg) Pour vérifier que ça fonctionne correctement
  for i in range(len(listes_données)):
      jeu = listes_données[:i]+listes_données[i+1:]
      test= listes_données[i]
  

    
    #a2=f'{jeu[2]}'
    
    
      modèle=global_loop(jeu,epochs)
    
      a2=f'{test}'
      dict_refed8[f'test {rep_geo[a2]}']= test_loop(modèle,test)
      with open('dict_refed8','wb') as f :
        pickle.dump(dict_refed8,f)
  return dict_refed8

liste_data=[R2018,R2019,R2020,L2018,L2019,L2020,T2018,T2019,T2020]

dict_refed_8=final_test(liste_data,100)
print(dict_refed_8)



