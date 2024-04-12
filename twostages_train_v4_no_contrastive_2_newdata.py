"""
二阶段，第一阶段 训练数据集是  0<vote< 10 五折，cv 是 对应的验证集，  第二阶段 10<vote<28  五折， cv是对应的验证集
正则项 在训练集里面加，验证集里面不加
无特征对齐 contrastive loss

https://www.kaggle.com/code/rafaelzimmermann1/no-ensemble-new-spectrograms-label-refine/notebook?scriptVersionId=169151032
https://www.kaggle.com/code/rafaelzimmermann1/hms-spectrogram-creation-using-gpu/notebook
https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/487110

"""
from nnspt.blocks.encoders import Encoder
import math
import os
import sys
import gc
from time import time
import random
sys.path.append('/root/autodl-tmp/kaggle-kl-div')
from kaggle_kl_div import score
import typing as tp
import torch
from torch import nn
import torch.nn.functional as F
from   torch.nn  import init
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.model_selection import KFold, GroupKFold
import timm
from pathlib import Path
import shutil
# from tqdm.notebook import tqdm
from tqdm import tqdm
from scipy.signal import butter, lfilter
import librosa
from typing import Dict, List, Union
import scipy.signal as scisig
class CFG:
    model_name ='tf_efficientnet_b0'#'mixnet_s'#'mixnet_xl'#'tf_efficientnet_b0'#'mixnet_s'# #'#"resnet34d"
    img_size = 512
    max_epoch = 10
    batch_size = 32
    lr = 3e-03
    weight_decay = 1.0e-02
    es_patience =  100
    seed = 1086
    enable_amp = True
    device = "cuda"
    stage_type = 'vote' # vote
    data_type = 'KEG' # K|E|R|KE|KR|ER|KER
    USE_PROCESSED = False
    hybrid= False
    # raw eeg
    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов
    out_samples = nsamples // 5
    
    
    map_features = [
        ("Fp1", "T3"),
        ("T3", "O1"),
        ("Fp1", "C3"),
        ("C3", "O1"),
        ("Fp2", "C4"),
        ("C4", "O2"),
        ("Fp2", "T4"),
        ("T4", "O2"),
    ]
    eeg_features = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]  # 'Fz', 'Cz', 'Pz' ,'F3', 'P3', 'F7', 'T5', 'Fz', 'Cz', 'Pz', 'F4', 'P4', 'F8', 'T6', 'EKG'
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'
    n_map_features = len(map_features)
    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    eeg_in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    random_close_zone = 0.0

    filter_order = 2
    
    # resnet gru
    kernels = [3, 5, 7, 9, 11]
    fixed_kernel_size = 5
    
    linear_layer_features = 304   # 1/5  Signal = 2_000
    ######
    bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
    rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    

    
def add_kl(data):
    import torch
    labels = data[TARGETS].values + 1e-5

    # compute kl-loss with uniform distribution by pytorch
    data['kl'] = torch.nn.functional.kl_div(
        torch.log(torch.tensor(labels)),
        torch.tensor([1 / 6] * 6),
        reduction='none'
    ).sum(dim=1).numpy()
    return data

train = pd.read_csv('/root/autodl-tmp/hms-harmful-brain-activity-classification/train.csv')
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

print('Train shape:', train.shape )
print('Targets', list(TARGETS))
# print(train.head())

if CFG.stage_type == 'kl':
    
    META = ['spectrogram_id','spectrogram_label_offset_seconds','patient_id','expert_consensus']
    train = train.groupby('eeg_id')[META+TARGETS
                           ].agg({**{m:'first' for m in META},**{t:'sum' for t in TARGETS}}).reset_index() 
    train[TARGETS] = train[TARGETS]/train[TARGETS].values.sum(axis=1,keepdims=True)
    train.columns = ['eeg_id','spec_id','offset','patient_id','target'] + TARGETS
    train = add_kl(train)
else:
    print('=======vote =======================')
    train["total_evaluators"] = train[TARGETS].sum(axis=1)
    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_label_offset_seconds': 'min',
        'spectrogram_label_offset_seconds': 'min',
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        'total_evaluators': 'mean',
        **{t: 'sum' for t in TARGETS},
        
    }

    train = train.groupby('eeg_id').agg(agg_functions).reset_index()
    y_data = train[TARGETS].values + 1e-5  # Regularization value
    train[TARGETS] = y_data / y_data.sum(axis=1, keepdims=True)
    train.columns = ['eeg_id','eeg_offset', 'offset','spec_id','patient_id','target','total_evaluators'] + TARGETS
    
    # META = ['spectrogram_id','spectrogram_label_offset_seconds','patient_id','expert_consensus','total_evaluators']
    # train = train.groupby('eeg_id')[META+TARGETS
    #                        ].agg({**{m:'first' for m in META},**{t:'sum' for t in TARGETS}}).reset_index() 
    
    # y_data = train[TARGETS].values + 1e-5  # Regularization value
    # train[TARGETS] = y_data / y_data.sum(axis=1, keepdims=True)
    # train.columns = ['eeg_id','spec_id','offset','patient_id','target','total_evaluators'] + TARGETS



    
# READ ALL SPECTROGRAMS
PATH = '/root/autodl-tmp/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')


spectrograms = None
all_eegs = None
all_raw_eegs = None
if CFG.data_type in ['K','KE','KR','KER']:  
    spectrograms = np.load('/root/autodl-tmp/brain-spectrograms/specs.npy',allow_pickle=True).item()


if CFG.data_type in ['E','KE','ER','KER']: 
    all_eegs = np.load('/root/autodl-tmp/eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()

if CFG.data_type in ['R','KR','ER','KER']: 
    if CFG.USE_PROCESSED:
        all_raw_eegs = np.load('/root/autodl-tmp/hms-eeg/eegs_processed.npy',allow_pickle=True).item()
    else:
        all_raw_eegs = np.load('/root/autodl-tmp/brain-eegs/eegs.npy',allow_pickle=True).item()

    

from audiomentations import AddGaussianNoise

transform = AddGaussianNoise(
    min_amplitude=0.001,
    max_amplitude=0.015,
    p=0.4
)
    
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x: y for y, x in TARS.items()}

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]



def spectrogram_from_eeg(parquet_path):
    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((100,300,4),dtype='float32')

    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
            # FILL NANS
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk+1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean()<1: x1 = np.nan_to_num(x1,nan=m)
            else: x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean()<1: x2 = np.nan_to_num(x2,nan=m)
            else: x2[:] = 0
                
            # COMPUTE PAIR DIFFERENCES
            x = x1 - x2

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//300, 
                  n_fft=1024, n_mels=100, fmin=0, fmax=20, win_length=128)
            
            # LOG TRANSFORM
            width = (mel_spec.shape[1]//30)*30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
          
    return img
    
def eeg_from_parquet(
    parquet_path: str
) -> np.ndarray:

    eeg = pd.read_parquet(parquet_path, columns=CFG.eeg_features)
    rows = len(eeg)
    offset = (rows- CFG.nsamples)//2
    eeg = eeg.iloc[offset:offset+ CFG.nsamples]
    data = np.zeros((CFG.nsamples,len(CFG.eeg_features)))
    for j,col in enumerate(CFG.eeg_features):
        
        # FILL NAN
        x = eeg[col].values.astype('float32')
        mean = np.nanmean(x)
        nan_percentage = np.isnan(x).mean()
        if nan_percentage < 1:
            x = np.nan_to_num(x, nan=mean)
        
        else: x[:] = 0
        
        data[:,j] = x

    return data


class EEGDataset(Dataset):
    
    def __init__(self, data, augment=False, mode='train', specs=None, eeg_specs=None,raw_eegs=None, data_type='both',bandpass_filter: Dict[str, Union[int, float]] = None, rand_filter: Dict[str, Union[int, float]] = None,downsample: int = None): 
        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.data_type = data_type
        self.bandpass_filter = bandpass_filter
        self.rand_filter = rand_filter
        self.downsample = downsample
        self.offset = None
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        X, y = self._generate_data(index)
        if self.augment:
            X = self._augment(X) 

        return X, y

    
    def _generate_data(self, index):
        if self.data_type == 'KE':
            X,y = self.generate_all_specs(index)
        elif self.data_type == 'KEG':
            X,y = self.generate_new_all_specs(index)
        elif self.data_type == 'E' or self.data_type == 'K':
            X,y = self.generate_specs(index)
        elif self.data_type == 'R':
            X,y = self.generate_raw(index)

        elif self.data_type in ['ER','KR']:
            X1,y = self.generate_specs(index)
            X2,y = self.generate_raw(index)
            X = (X1,X2)
        elif self.data_type in ['KER']:
            X1,y = self.generate_all_specs(index)
            X2,y = self.generate_raw(index)
            X = (X1,X2)
        return X,y
        
    def generate_new_all_specs(self, index):
        X = np.zeros((512,512,3),dtype='float32')
        y = np.zeros((6,),dtype='float32')
        row = self.data.iloc[index]
        spec_offset = int(row['offset'])
        eeg_offset = int(row['eeg_offset'])
        
        file_path = f'/root/autodl-tmp/3_diff_time_specs/images/{row.eeg_id}_{spec_offset}_{eeg_offset}.npz'
        data = np.load(file_path)
        eeg_data = data['final_image']
        eeg_data_expanded = np.repeat(eeg_data[:, :, np.newaxis], 3, axis=2)
        X = eeg_data_expanded
        if self.mode!='test':
            y[:] = row[TARGETS]
        return X,y
    def generate_all_specs(self, index):
        X = np.zeros((512,512,3),dtype='float32')
        y = np.zeros((6,),dtype='float32')
    
        row = self.data.iloc[index]
        if self.mode == 'test': 
            offset = 0
        else:
            offset = int(row.offset/2)
        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

            
        imgs = [spec[offset:offset+300,k*100:(k+1)*100].T for k in [0,2,1,3]] # to match kaggle with eeg
        img = np.stack(imgs,axis=-1)
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img,np.exp(-4),np.exp(8))
        img = np.log(img)

        # STANDARDIZE PER IMAGE
        img = np.nan_to_num(img, nan=0.0)    
            
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[0_0+56:100+56,:256,0] = img[:,22:-22,0] # LL_k
        X[100+56:200+56,:256,0] = img[:,22:-22,2] # RL_k
        X[0_0+56:100+56,:256,1] = img[:,22:-22,1] # LP_k
        X[100+56:200+56,:256,1] = img[:,22:-22,3] # RP_k
        X[0_0+56:100+56,:256,2] = img[:,22:-22,2] # RL_k
        X[100+56:200+56,:256,2] = img[:,22:-22,1] # LP_k
        
        X[0_0+56:100+56,256:,0] = img[:,22:-22,0] # LL_k
        X[100+56:200+56,256:,0] = img[:,22:-22,2] # RL_k
        X[0_0+56:100+56,256:,1] = img[:,22:-22,1] # LP_k
        X[100+56:200+56,256:,1] = img[:,22:-22,3] # RP_K

        # EEG
        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200+56:300+56,:256,0] = img[:,22:-22,0] # LL_e
        X[300+56:400+56,:256,0] = img[:,22:-22,2] # RL_e
        X[200+56:300+56,:256,1] = img[:,22:-22,1] # LP_e
        X[300+56:400+56,:256,1] = img[:,22:-22,3] # RP_e
        X[200+56:300+56,:256,2] = img[:,22:-22,2] # RL_e
        X[300+56:400+56,:256,2] = img[:,22:-22,1] # LP_e
        
        X[200+56:300+56,256:,0] = img[:,22:-22,0] # LL_e
        X[300+56:400+56,256:,0] = img[:,22:-22,2] # RL_e
        X[200+56:300+56,256:,1] = img[:,22:-22,1] # LP_e
        X[300+56:400+56,256:,1] = img[:,22:-22,3] # RP_e
        if self.mode!='test':
            y[:] = row[TARGETS]
            

        return X,y
    
    def generate_specs(self, index):
        X = np.zeros((512,512,3),dtype='float32')
        y = np.zeros((6,),dtype='float32')
        
        row = self.data.iloc[index]
        if self.mode=='test': 
            offset = 0
        else:
            offset = int(row.offset/2)
            
        if self.data_type in ['E','ER']:
            img = self.eeg_specs[row.eeg_id]
        elif self.data_type in ['K','KR']:
            spec = self.specs[row.spec_id]
            imgs = [spec[offset:offset+300,k*100:(k+1)*100].T for k in [0,2,1,3]] # to match kaggle with eeg
            img = np.stack(imgs,axis=-1)
            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img,np.exp(-4),np.exp(8))
            img = np.log(img)
            
            # STANDARDIZE PER IMAGE
            img = np.nan_to_num(img, nan=0.0)     
            
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        
        X[0_0+56:100+56,:256,0] = img[:,22:-22,0]
        X[100+56:200+56,:256,0] = img[:,22:-22,2]
        X[0_0+56:100+56,:256,1] = img[:,22:-22,1]
        X[100+56:200+56,:256,1] = img[:,22:-22,3]
        X[0_0+56:100+56,:256,2] = img[:,22:-22,2]
        X[100+56:200+56,:256,2] = img[:,22:-22,1]
        
        X[0_0+56:100+56,256:,0] = img[:,22:-22,0]
        X[100+56:200+56,256:,0] = img[:,22:-22,1]
        X[0_0+56:100+56,256:,1] = img[:,22:-22,2]
        X[100+56:200+56,256:,1] = img[:,22:-22,3]
        
        X[200+56:300+56,:256,0] = img[:,22:-22,0]
        X[300+56:400+56,:256,0] = img[:,22:-22,1]
        X[200+56:300+56,:256,1] = img[:,22:-22,2]
        X[300+56:400+56,:256,1] = img[:,22:-22,3]
        X[200+56:300+56,:256,2] = img[:,22:-22,3]
        X[300+56:400+56,:256,2] = img[:,22:-22,2]
        
        X[200+56:300+56,256:,0] = img[:,22:-22,0]
        X[300+56:400+56,256:,0] = img[:,22:-22,2]
        X[200+56:300+56,256:,1] = img[:,22:-22,1]
        X[300+56:400+56,256:,1] = img[:,22:-22,3]
        
        if self.mode!='test':
            y[:] = row[TARGETS]
        
        return X,y
    

    
    def generate_raw(self,index):        
        X = np.zeros((CFG.out_samples, CFG.eeg_in_channels),dtype='float32')
        y = np.zeros((6,),dtype='float32')
        

        row = self.data.iloc[index]
        data = self.raw_eegs[row.eeg_id]
            
        if CFG.nsamples != CFG.out_samples:
            if self.mode != "train":
                offset = (CFG.nsamples - CFG.out_samples) // 2
            else:
                #offset = random.randint(0, CFG.nsamples - CFG.out_samples)                
                offset = ((CFG.nsamples - CFG.out_samples) * random.randint(0, 1000)) // 1000
            data = data[offset:offset+CFG.out_samples,:]
                
        for i, (feat_a, feat_b) in enumerate(CFG.map_features):
            if self.mode == "train" and CFG.random_close_zone > 0 and random.uniform(0.0, 1.0) <= CFG.random_close_zone:
                continue
                
            diff_feat = (
                data[:, CFG.feature_to_index[feat_a]]
                - data[:, CFG.feature_to_index[feat_b]]
            )  # Size=(10000,)
                            
            
            if not self.bandpass_filter is None:
                diff_feat = self.butter_bandpass_filter(
                    diff_feat,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )
                
                
            if (
                self.mode == "train"
                and not self.rand_filter is None
                and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                diff_feat = self.butter_bandpass_filter(
                    diff_feat,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )

            X[:, i] = diff_feat
            
        n = CFG.n_map_features
        if len(CFG.freq_channels) > 0:
            for i in range(CFG.n_map_features):
                diff_feat = X[:, i]
                for j, (lowcut, highcut) in enumerate(CFG.freq_channels):
                    band_feat = self.butter_bandpass_filter(
                        diff_feat, lowcut, highcut, CFG.sampling_rate, order=CFG.filter_order,  # 6
                    )
                    X[:, n] = band_feat
                    n += 1
        
        for spml_feat in CFG.simple_features:
            feat_val = data[:, CFG.feature_to_index[spml_feat]]
            
            if not self.bandpass_filter is None:
                feat_val = self.butter_bandpass_filter(
                    feat_val,
                    self.bandpass_filter["low"],
                    self.bandpass_filter["high"],
                    CFG.sampling_rate,
                    order=self.bandpass_filter["order"],
                )
                
            if (
                self.mode == "train"
                and not self.rand_filter is None
                and random.uniform(0.0, 1.0) <= self.rand_filter["probab"]
            ):
                lowcut = random.randint(
                    self.rand_filter["low"], self.rand_filter["high"]
                )
                highcut = lowcut + self.rand_filter["band"]
                feat_val = butter_bandpass_filter(
                    feat_val,
                    lowcut,
                    highcut,
                    CFG.sampling_rate,
                    order=self.rand_filter["order"],
                )
            X[:, n] = feat_val
            n += 1
            
        # STANDARDIZE
        X = np.clip(X,-1024,1024)
        X = np.nan_to_num(X, nan=0) / 32.0
            
        # BUTTER LOW-PASS FILTER
        X = self.butter_lowpass_filter(X, order=CFG.filter_order)
        # Downsample
        if self.downsample is not None:
            X = X[::self.downsample,:]
        
        if self.mode!='test':
            y[:] = row[TARGETS]
                
        return X,y

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def butter_lowpass_filter(
        self, data, cutoff_freq=20, sampling_rate=CFG.sampling_rate, order=4
    ):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data, axis=0)
        return filtered_data
    
    def _random_transform(self, img):
        composition = albu.Compose([
#             albu.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            albu.HorizontalFlip(p=0.4),
            # albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']
            
    def _augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        return img_batch


    
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
    

class mlp(nn.Module):
    def __init__(self, in_features, hidden_size,num_cls):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_size)
        self.l2 = nn.Linear(in_features, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_cls)
        self.atc = nn.SiLU()

    def forward(self, x):
        return self.l3(self.atc(self.l2(x))*self.l1(x))
    
class HMSHBACSpecModel(nn.Module):

    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            in_channels: int,
            num_classes: int,
            hybrid=False
        ):
        super().__init__()
        self.hybrid= hybrid
        # self.model = timm.create_model(
        #     model_name=model_name, pretrained=False, checkpoint_path='./mixnet_s.bin', in_chans=in_channels)
        self.model = timm.create_model(
            model_name=model_name, pretrained=True, in_chans=in_channels)

        in_features = self.model.classifier.in_features
        self.in_features = in_features
        self.model.classifier = nn.Identity()
        # if not self.hybrid:
        self.mlp = mlp(in_features, in_features*2,num_classes)
        # self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):   
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)
        features = self.model(x) # torch.Size([32, 1536])

        out = self.mlp(features)
        if self.hybrid:
            return features, out
        else:
            return out
    
# class HMSHBACSpecModel(nn.Module):

#     def __init__(
#             self,
#             model_name: str,
#             pretrained: bool,
#             in_channels: int,
#             num_classes: int,
#             hybrid=False
#         ):
#         super().__init__()
#         self.hybrid= hybrid
#         self.model = timm.create_model(
#             model_name=model_name, pretrained=False, checkpoint_path='./efvit_b0.bin', in_chans=in_channels)
        
#         in_features = 1024
#         self.in_features = in_features
#         self.model.head.classifier = nn.Identity()
#         # if not self.hybrid:
#         self.mlp = mlp(in_features, in_features*2,num_classes)
#     def forward(self, x):   
#         x = torch.Tensor(x)
#         x = x.permute(0, 3, 1, 2)
#         features = self.model(x)
#         out = self.mlp(features)
#         if self.hybrid:
#             return features, out
#         else:
#             return out



class ResNet_1D_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        downsampling,
        dilation=1,
        groups=1,
        dropout=0.0,
    ):
        super(ResNet_1D_Block, self).__init__()
    
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        
        
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out
class ResnetGruNet(nn.Module):
    def __init__(
        self,
        kernels,
        in_channels,
        fixed_kernel_size,
        num_classes,
        linear_layer_features,
        dilation=1,
        groups=1,
        hybrid=False
    ):
        super(ResnetGruNet, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.hybrid = hybrid
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.ReLU()
        # self.relu_2 = nn.ReLU()
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            # dropout=0.2,
        )
        # if not hybrid:
        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
        self,
        kernel_size,
        stride,
        dilation=1,
        groups=1,
        blocks=9,
        padding=0,
        dropout=0.0,
    ):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)

        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out) # torch.Size([32, 24, 9])
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1) # torch.Size([32, 48])
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]  # torch.Size([32, 256])
        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        x = x.float()
        new_out = self.extract_features(x)

        
        result = self.fc(new_out)
        if self.hybrid:
            return new_out, result
        else:
            return result



        
    
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    
# class HMSHBACSpecModel(nn.Module):

#     def __init__(
#             self,
#             model_name: str,
#             pretrained: bool,
#             in_channels: int,
#             num_classes: int,
#             flag = 'train'
#         ):
#         super().__init__()
#         self.flag = flag
#         self.backbone = timm.create_model(
#             model_name=model_name, pretrained=pretrained, in_chans=in_channels)
#         feature_info = self.backbone.feature_info
#         self.block_out_idx = [1, 2, 4]

#         self.aux_block1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[1]["num_chs"])
#         )
#         self.aux_block2 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[2]["num_chs"])
#         )
#         self.aux_block4 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[3]["num_chs"])
#         )

#         self.aux_linear1 = nn.Linear(feature_info[1]["num_chs"], num_classes)
#         self.aux_linear2 = nn.Linear(feature_info[2]["num_chs"], num_classes)
#         self.aux_linear4 = nn.Linear(feature_info[3]["num_chs"], num_classes)
#         self.num_features = (
#             self.backbone.num_features
#             + feature_info[1]["num_chs"]
#             + feature_info[2]["num_chs"]
#             + feature_info[3]["num_chs"]
#         )
#         self.linear = nn.Linear(self.num_features, num_classes)
#         # self.mlp = mlp(self.num_features, self.num_features*2,num_classes)
#     def forward_features(self, x):
#         x = self.backbone.conv_stem(x)
#         x = self.backbone.bn1(x)
#         features = []
#         for i, b in enumerate(self.backbone.blocks):
#             x = b(x)
#             if i in self.block_out_idx:
#                 features.append(x)

#         features[0] = self.aux_block1(features[0])
#         features[1] = self.aux_block2(features[1])
#         features[2] = self.aux_block4(features[2])
#         features.append(self.backbone.global_pool(self.backbone.bn2(self.backbone.conv_head(x))))
#         return features


#     def forward(self, x):   
#         x = torch.Tensor(x)
#         x = x.permute(0, 3, 1, 2)
#         features = self.forward_features(x)
#         if self.flag ==  'train':
#             out1 = self.aux_linear1(features[0])
#             out2 = self.aux_linear2(features[1])
#             out4 = self.aux_linear4(features[2])
#             features = torch.cat(features, dim=1)
#             out = self.linear(features)
#             return out, out1, out2 ,out4
#         else:
#             features = torch.cat(features, dim=1)
#             out = self.linear(features)
#             return out


# class EEGEfficientnetClassifier(torch.nn.Module):
#     def __init__(
#         self, 
#         in_channels,
#         nclasses, 
#         encoder='timm-efficientnetv2-b1',
#         hybrid=True
#     ):
#         """
#             :args:
#                 nleads (int): the number of leads of the inputed ECGs
#                 nclasses (int): the number of predicted classes
#         """
#         super().__init__()
#         self.hybrid = hybrid
            
#         self.encoder = Encoder(in_channels=in_channels, depth=5, name=encoder)
#         self.avg = torch.nn.AdaptiveAvgPool1d(1)
#         self.flatten = torch.nn.Flatten()
#         self.eeg_features_num = self.encoder.out_channels[-1]+256
#         self.rnn = nn.GRU(
#             input_size=in_channels,
#             hidden_size=128,
#             num_layers=1,
#             bidirectional=True,
#             # dropout=0.2,
#         )
#         self.fc = torch.nn.Linear(self.eeg_features_num, nclasses)

#     def forward(self, x):
#         x = x.permute(0, 2, 1).float()
#         rnn_out, _ = self.rnn(x.permute(0, 2, 1))
#         new_rnn_h = rnn_out[:, -1, :]  # torch.Size([32, 256])

        
#         f = self.encoder(x)
#         x = self.avg(f[-1])
#         features = self.flatten(x)
#         new_feature = torch.cat([features, new_rnn_h], dim=1)
#         output = self.fc(new_feature)
#         if self.hybrid:
#             return new_feature, output
#         else:
#             return output
class MulitModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        spec_in_channels: int,
        num_classes: int,
        kernels,
        eeg_in_channels,
        fixed_kernel_size,
        linear_layer_features,
        hybrid=True,
    ):
        super(MulitModel, self).__init__()
        self.spec_model = HMSHBACSpecModel(
        model_name=model_name, pretrained=pretrained, num_classes=num_classes, in_channels=spec_in_channels, hybrid=hybrid)
        spec_features_num = self.spec_model.in_features
        self.eeg_model =  ResnetGruNet(
            kernels=kernels,
            in_channels=eeg_in_channels,
            fixed_kernel_size=fixed_kernel_size,
            num_classes=num_classes,
            linear_layer_features=linear_layer_features,
            hybrid=True,
        )
        
        self.fc = nn.Linear(in_features=linear_layer_features + spec_features_num, out_features=num_classes)
        # self.mlp = mlp(linear_layer_features + spec_features_num, (linear_layer_features + spec_features_num)*2,num_classes)
    def forward(self, x1,x2):
        spec_feature,pred1 = self.spec_model(x1)
        eeg_feature, pred2 = self.eeg_model(x2)
        new_features = torch.cat([spec_feature,eeg_feature],dim=1)
        output = self.fc(new_features)
      
        
        return output, pred1, pred2
    
class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss
    
class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.cpu().numpy())
        self.label_list.append(t.cpu().numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric
    
    
def set_random_seed(seed: int = 42):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)
    
    
def train_one_fold(CFG, fold,train_index, valid_index, output_path,stage='first'):
    """Main"""
    set_random_seed(CFG.seed)
    device = torch.device(CFG.device)
    if CFG.data_type in ['K','E','KE','KEG']:
        print('-----------SpecModel----------------')
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=3)
        
    elif CFG.data_type in ['KR','ER','KER']:
        print('-----------muLITMODEL----------------')
        model = MulitModel(
            model_name = CFG.model_name,
            pretrained=True,
            num_classes=6,
            spec_in_channels = 3,
            kernels=CFG.kernels,
            eeg_in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            linear_layer_features=CFG.linear_layer_features,               
        )
    elif CFG.data_type in ['R']:
        # model = WaveNetModel()
        model = ResnetGruNet(
            kernels=CFG.kernels,
            in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            num_classes=6,
            linear_layer_features=CFG.linear_layer_features,        
        )
        print('-------------ResnetGruNet---------------')
    model.to(device)
    
    if stage !='first':
        train_data, valid_data = sec_train.iloc[train_index], sec_train.iloc[valid_index]
        train_ds = EEGDataset(data=train_data, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type, bandpass_filter=CFG.bandpass_filter,rand_filter=CFG.rand_filter)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.batch_size, num_workers=10,drop_last=True)
  
        valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type,bandpass_filter=CFG.bandpass_filter)
        val_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=10)
        
        model_path = './'+ f"best_model_fold{fold}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('init success')
    else:
        train_data, valid_data = first_train.iloc[train_index], first_train.iloc[valid_index]
        train_ds = EEGDataset(data=train_data, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type, bandpass_filter=CFG.bandpass_filter,rand_filter=CFG.rand_filter)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.batch_size, num_workers=10)
        valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type,bandpass_filter=CFG.bandpass_filter)
        val_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=10)
    

    print(f'### Train size: {len(train_index)}, Valid size: {len(valid_index)}')
    print('#'*25)
    

    
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, epochs=CFG.max_epoch,
        pct_start=0.0, steps_per_epoch=len(train_loader),
        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    )

    loss_func = KLDivLossWithLogits()
    loss_func.to(device)
    loss_func_val = KLDivLossWithLogitsForVal()
    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)
    
    best_val_loss = 1.0e+09
    best_epoch = 0
    train_loss = 0
    for epoch in range(1, CFG.max_epoch + 1):
        epoch_start = time()
        model.train()
        for batch in tqdm(train_loader):
#             batch = to_device(batch, device)
            if CFG.hybrid:
                (x1, x2), t = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                t = t.to(device)
            else:
                x, t = batch
                x = x.to(device)
                t = t.to(device)
            optimizer.zero_grad()
            with amp.autocast(use_amp):
                if CFG.hybrid:
                    y, y1, y2= model(x1,x2)
                    loss = loss_func(y, t)
                    loss1 = loss_func(y1, t)
                    loss2 = loss_func(y2, t)
                    loss3 = loss_func(y*0.5 + y1*0.25 + y2*0.25, t)   
                    loss = loss + loss1 + loss2 + loss3
                else:
                    y = model(x)
                    loss = loss_func(y, t)

    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
            
        model.eval()
        for batch in tqdm(val_loader):
#             batch = to_device(batch, device)
            if CFG.hybrid:
                (x1, x2), t = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                t = t.to(device)
            else:
                x, t = batch
                x = x.to(device)
                t = t.to(device)
            with torch.no_grad(), amp.autocast(use_amp):
                if CFG.hybrid:
                    y, y1, y2 = model(x1,x2)
                    y = y*0.5 + y1*0.25 + y2*0.25
                else:
                    y = model(x)
            
            y = y.detach().cpu().to(torch.float32)
            loss_func_val(y, t)
        val_loss = loss_func_val.compute() 
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            # print("save model")
            torch.save(model.state_dict(), str(output_path / f'snapshot_epoch_{epoch}.pth'))
        
        elapsed_time = time() - epoch_start
        print(
            f"[epoch {epoch}] train loss: {train_loss: .6f}, val loss: {val_loss: .6f}, elapsed_time: {elapsed_time: .3f}")
        
        if epoch - best_epoch > CFG.es_patience:
            print("Early Stopping!")
            break
            
        train_loss = 0
            
    return fold, best_epoch, best_val_loss

def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if CFG.hybrid:
                (x1, x2), t = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                t = t.to(device)
                y, y1, y2= model(x1,x2)
                y = y*0.5 + y1*0.25 + y2*0.25
            else:
                x, t = batch
                x = x.to(device)
                t = t.to(device)
                y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr

if CFG.stage_type == 'kl':
    sec_train = train[train['kl']<5.5]
else:
    sec_train = train[train['total_evaluators']>=10]
    first_train = train[train['total_evaluators']<10]
gkf = GroupKFold(n_splits=5)
# 第一阶段
score_list = []
for i, (train_index, valid_index) in enumerate(gkf.split(first_train, first_train.target, first_train.patient_id)):   
    print('#'*25)
    print(f'###  Fold {i}')
    
    output_path = Path(f"fold{i}")
    output_path.mkdir(exist_ok=True)
    print(f"[fold{i}]")     
    score_list.append(train_one_fold(CFG, i,train_index, valid_index, output_path))

print('============first stage===============')
print(score_list)

best_log_list = []
for (fold_id, best_epoch, _) in score_list:
    
    exp_dir_path = Path(f"fold{fold_id}")
    best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
    copy_to = f"./best_model_fold{fold_id}.pth"
    shutil.copy(best_model_path, copy_to)
    
    for p in exp_dir_path.glob("*.pth"):
        p.unlink()
        

        
device = torch.device(CFG.device)


all_oof = []
all_true = []
valid_loaders = []

for i, (train_index, valid_index) in enumerate(gkf.split(first_train, first_train.target, first_train.patient_id)):  
    print('#'*25)
    print(f'### Fold {i+1}')
    valid_data = first_train.iloc[valid_index]
    
    valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type,bandpass_filter=CFG.bandpass_filter)
    valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=3)


    print(f'###  Valid size: {len(valid_index)}')
    print('#'*25)
    
    # # get model
    model_path = f"./best_model_fold{i}.pth"
    if CFG.data_type in ['K','E','KE','KEG']:
        print('-----------SpecModel----------------')
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=3)
        
    elif CFG.data_type in ['KR','ER','KER']:
        model = MulitModel(
            model_name = CFG.model_name,
            pretrained=True,
            num_classes=6,
            spec_in_channels = 3,
            kernels=CFG.kernels,
            eeg_in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            linear_layer_features=CFG.linear_layer_features,               
        )
    elif CFG.data_type in ['R']:
        model = ResnetGruNet(
            kernels=CFG.kernels,
            in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            num_classes=6,
            linear_layer_features=CFG.linear_layer_features,        
        )
        print('-------------ResnetGru---------------')
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # # inference
    val_pred = run_inference_loop(model, valid_loader, device)
    valid_loaders.append(valid_loader)
    all_true.append(first_train.iloc[valid_index][TARGETS].values)
    all_oof.append(val_pred)
    del model
    gc.collect()

    
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print(f'CV Score KL-Div for {CFG.model_name} =',cv)
        
        

# 第二阶段
score_list = []
for i, (train_index, valid_index) in enumerate(gkf.split(sec_train, sec_train.target, sec_train.patient_id)):  
    print('#'*25)
    print(f'###  Fold {i}')
    
    output_path = Path(f"fold{i}")
    output_path.mkdir(exist_ok=True)
    print(f"[fold{i}]")     
    score_list.append(train_one_fold(CFG, i,train_index, valid_index, output_path, stage='sec'))
    
print('=========sec stage============')
print(score_list)
best_log_list = []
for (fold_id, best_epoch, _) in score_list:
    
    exp_dir_path = Path(f"fold{fold_id}")
    best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
    copy_to = f"./best_model_fold{fold_id}.pth"
    shutil.copy(best_model_path, copy_to)
    
    for p in exp_dir_path.glob("*.pth"):
        p.unlink()
        



device = torch.device(CFG.device)


all_oof = []
all_true = []
valid_loaders = []
for i, (train_index, valid_index) in enumerate(gkf.split(sec_train, sec_train.target, sec_train.patient_id)):   
    print('#'*25)
    print(f'### Fold {i+1}')
    valid_data = sec_train.iloc[valid_index]
    
    valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type,bandpass_filter=CFG.bandpass_filter)
    valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=3)


    print(f'###  Valid size: {len(valid_index)}')
    print('#'*25)
    
    # # get model
    model_path = f"./best_model_fold{i}.pth"
    if CFG.data_type in ['K','E','KE','KEG']:
        print('-----------SpecModel----------------')
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=3)
        
    elif CFG.data_type in ['KR','ER','KER']:
        model = MulitModel(
            model_name = CFG.model_name,
            pretrained=True,
            num_classes=6,
            spec_in_channels = 3,
            kernels=CFG.kernels,
            eeg_in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            linear_layer_features=CFG.linear_layer_features,               
        )
    elif CFG.data_type in ['R']:
        model = ResnetGruNet(
            kernels=CFG.kernels,
            in_channels=CFG.eeg_in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            num_classes=6,
            linear_layer_features=CFG.linear_layer_features,        
        )
        print('-------------ResnetGru---------------')
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # # inference
    val_pred = run_inference_loop(model, valid_loader, device)
    valid_loaders.append(valid_loader)
    all_true.append(sec_train.iloc[valid_index][TARGETS].values)
    all_oof.append(val_pred)
    del model
    gc.collect()

    
    
all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print(f'CV Score KL-Div for {CFG.model_name} =',cv)



