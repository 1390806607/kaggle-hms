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
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import pandas as pd, numpy as np
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


class CFG:
    model_name = 'tf_efficientnet_b0_ns'#"resnet34d"
    img_size = 512
    max_epoch = 10
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience =  10
    seed = 1086
    enable_amp = True
    device = "cuda"
    # USE_KAGGLE_SPECTROGRAMS = True
    # USE_EEG_SPECTROGRAMS = True
    # READ_SPEC_FILES = False
    # READ_EEG_SPEC_FILES = False
    # READ_RAW_EEGS = False
    VER = 5
    data_type = 'R' # K|E|R|KE|KR|ER|KER
    USE_PROCESSED = False
    
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
META = ['spectrogram_id','spectrogram_label_offset_seconds','patient_id','expert_consensus']
train = train.groupby('eeg_id')[META+TARGETS
                       ].agg({**{m:'first' for m in META},**{t:'sum' for t in TARGETS}}).reset_index() 
train[TARGETS] = train[TARGETS]/train[TARGETS].values.sum(axis=1,keepdims=True)
train.columns = ['eeg_id','spec_id','offset','patient_id','target'] + TARGETS
train = add_kl(train)
print(train.head(5).to_string())




# READ ALL SPECTROGRAMS
PATH = '/root/autodl-tmp/hms-harmful-brain-activity-classification/train_spectrograms/'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')


spectrograms = None
all_eegs = None
all_raw_eegs = None
if CFG.data_type in ['K','KE','KR','KER']:  
    spectrograms = np.load('/root/autodl-tmp/brain-spectrograms/specs.npy',allow_pickle=True).item()
# if CFG.READ_SPEC_FILES:    
#     spectrograms = {}
#     for i,f in enumerate(files):
#         if i % 100 == 0:
#             print(i, ', ', end='')
#         tmp = pd.read_parquet(f'{PATH}{f}')
#         name = int(f.split('.')[0])
#         spectrograms[name] = tmp.iloc[:,1:].values
# else:
#     spectrograms = np.load('/root/autodl-tmp/brain-spectrograms/specs.npy',allow_pickle=True).item()
    

if CFG.data_type in ['E','KE','ER','KER']: 
    all_eegs = np.load('/root/autodl-tmp/eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()
# if CFG.READ_EEG_SPEC_FILES:
#     all_eegs = {}
#     for i,e in enumerate(train.eeg_id.values):
#         if i % 100 == 0:
#             print(i, ', ', end='')
#         x = np.load(f'/root/autodl-tmp/brain-eeg-spectrograms/EEG_Spectrograms/{e}.npy')
#         all_eegs[e] = x
# else:
# #     all_eegs = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()
#     all_eegs = np.load('/root/autodl-tmp/eeg-spectrograms/eeg_specs.npy',allow_pickle=True).item()

if CFG.data_type in ['R','KR','ER','KER']: 
    if CFG.USE_PROCESSED:
        all_raw_eegs = np.load('/root/autodl-tmp/hms-eeg/eegs_processed.npy',allow_pickle=True).item()
    else:
        all_raw_eegs = np.load('/root/autodl-tmp/hms-eeg/eegs.npy',allow_pickle=True).item()
# if CFG.READ_RAW_EEGS:
#     all_raw_eegs = np.load('/root/autodl-tmp/brain-eegs/eegs.npy',allow_pickle=True).item()
# else:
#     all_raw_eegs = None
    

from audiomentations import AddGaussianNoise

transform = AddGaussianNoise(
    min_amplitude=0.001,
    max_amplitude=0.015,
    p=0.4
)
    
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x: y for y, x in TARS.items()}
FEATS2 = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
FEAT2IDX = {x:y for x,y in zip(FEATS2,range(len(FEATS2)))}
FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

class EEGDataset(Dataset):
    
    def __init__(self, data, augment=False, mode='train', specs=None, eeg_specs=None, raw_eegs=None, data_type='both'): 
        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.data_type = data_type
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        X, y = self._generate_data(index)
        if self.augment:
            X = self._augment(X) 
#         if self.mode == 'train':
        return X, y
#         else:
#             return X
    
#     def __getitems__(self, index):
#         print(index)
#         X, y = self._generate_data(index)
#         if self.augment:
#             X = self._augment(X) 
#         return list(zip(X, y))
    
    def _generate_data(self, index):
        if self.data_type == 'KE':
            X,y = self.generate_all_specs(index)
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

            
#         if self.mode == 'train':
#             t1 = 1
#             for i in range(len(eeg.shape)):
#                 t1 *= eeg.shape[i] 
#             t2 = 1
#             for i in range(len(spec.shape)):
#                 t2 *= spec.shape[i]

#             eeg = transform(eeg, sample_rate=t1 // 2)
#             spec = np.squeeze(transform(spec[None,:,:], sample_rate=t2 // 2),0)
            
        # imgs = [spec[offset:offset+300,k*100:(k+1)*100].T for k in range(4)]
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
            # imgs = [spec[offset:offset+300,k*100:(k+1)*100].T for k in range(4)]
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
        if CFG.USE_PROCESSED and self.mode!='test':
            X = np.zeros((2_000,8),dtype='float32')
            y = np.zeros((6,),dtype='float32')
            row = self.data.iloc[index]
            X = self.raw_eegs[row.eeg_id]
            y[:] = row[TARGETS]
            return X,y
        
        
        X = np.zeros((10_000,8),dtype='float32')
        y = np.zeros((6,),dtype='float32')
        
        row = self.data.iloc[index]
        eeg = self.raw_eegs[row.eeg_id]
            
        # FEATURE ENGINEER
        X[:,0] = eeg[:,FEAT2IDX['Fp1']] - eeg[:,FEAT2IDX['T3']]
        X[:,1] = eeg[:,FEAT2IDX['T3']] - eeg[:,FEAT2IDX['O1']]
            
        X[:,2] = eeg[:,FEAT2IDX['Fp1']] - eeg[:,FEAT2IDX['C3']]
        X[:,3] = eeg[:,FEAT2IDX['C3']] - eeg[:,FEAT2IDX['O1']]
            
        X[:,4] = eeg[:,FEAT2IDX['Fp2']] - eeg[:,FEAT2IDX['C4']]
        X[:,5] = eeg[:,FEAT2IDX['C4']] - eeg[:,FEAT2IDX['O2']]
            
        X[:,6] = eeg[:,FEAT2IDX['Fp2']] - eeg[:,FEAT2IDX['T4']]
        X[:,7] = eeg[:,FEAT2IDX['T4']] - eeg[:,FEAT2IDX['O2']]
            
        # STANDARDIZE
        X = np.clip(X,-1024,1024)
        X = np.nan_to_num(X, nan=0) / 32.0
            
        # BUTTER LOW-PASS FILTER
        X = self.butter_lowpass_filter(X)
        # Downsample
        X = X[::5,:]
        
        if self.mode!='test':
            y[:] = row[TARGETS]
                
        return X,y
    
    
    def butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
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
    
def eeg_from_parquet(parquet_path):

    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows-10_000)//2
    eeg = eeg.iloc[offset:offset+10_000]
    data = np.zeros((10_000,len(FEATS2)))
    for j,col in enumerate(FEATS2):
        
        # FILL NAN
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
        else: x[:] = 0
        
        data[:,j] = x

    return data
    
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
        ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, in_chans=in_channels)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.mlp = mlp(in_features, in_features*2,num_classes)
    def forward(self, x):   
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)
        features = self.model(x)
        out = self.mlp(features)
        return out
    
class Wave_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int = 3):
        """
        WaveNet building block.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rates: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
        
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))
        
        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x) 
            res = res + x
        return res
    
class WaveNet(nn.Module):
    def __init__(self, input_channels: int = 1, kernel_size: int = 3):
        super(WaveNet, self).__init__()
        self.model = nn.Sequential(
                Wave_Block(input_channels, 8, 6, kernel_size),
                Wave_Block(8, 16, 6, kernel_size),
                Wave_Block(16, 32, 6, kernel_size),
                Wave_Block(32, 64, 6, kernel_size) 
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        output = self.model(x)
        return output


class WaveNetModel(nn.Module):
    def __init__(self):
        super(WaveNetModel, self).__init__()
        self.model = WaveNet()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = 0.0
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 6)
        )
        
    def forward(self, x: torch.Tensor):
        """
        Forwward pass.
        """
        x = x.float()
        x1 = self.model(x[:, :, 0:1])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(-1)
        x2 = self.model(x[:, :, 1:2])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(-1)
        z1 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 2:3])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(-1)
        x2 = self.model(x[:, :, 3:4])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(-1)
        z2 = torch.mean(torch.stack([x1, x2]), dim=0)
        
        x1 = self.model(x[:, :, 4:5])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(-1)
        x2 = self.model(x[:, :, 5:6])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(-1)
        z3 = torch.mean(torch.stack([x1, x2]), dim=0)
        
        x1 = self.model(x[:, :, 6:7])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze(-1)
        x2 = self.model(x[:, :, 7:8])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze(-1)
        z4 = torch.mean(torch.stack([x1, x2]), dim=0)
        
        y = torch.cat([z1, z2, z3, z4], dim=1)
        y = self.head(y)
        
        return y
    
    
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
    if CFG.data_type in ['K','E','KE']:
        print('-----------SpecModel----------------')
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=3)
        
    elif CFG.data_type in ['KR','ER','KER']:
        pass
    elif CFG.data_type in ['R']:
        model = WaveNetModel()
        print('-------------WaveNet---------------')
    model.to(device)
    
    
    train_data, valid_data = train.iloc[train_index], train.iloc[valid_index]
    train_ds = EEGDataset(data=train_data, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.batch_size, num_workers=10)
    valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type)
    val_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=10)
    
    if stage !='first':
        train_data= train_data[train_data['kl']<5.5]
        train_ds = EEGDataset(data=train_data, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type)
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.batch_size, num_workers=10)
        model_path = './'+ f"best_model_fold{fold}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('init success')
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
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            with amp.autocast(use_amp):
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
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            with torch.no_grad(), amp.autocast(use_amp):
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


score_list = []

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    print('#'*25)
    print(f'###  Fold {i}')
    
    output_path = Path(f"fold{i}")
    output_path.mkdir(exist_ok=True)
    print(f"[fold{i}]")     
    score_list.append(train_one_fold(CFG, i,train_index, valid_index, output_path))
    # score_list.append(train_one_fold(CFG, i,train_index, valid_index, output_path, stage='sec'))
    
    
print(score_list)

best_log_list = []
for (fold_id, best_epoch, _) in score_list:
    
    exp_dir_path = Path(f"fold{fold_id}")
    best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
    copy_to = f"./best_model_fold{fold_id}.pth"
    shutil.copy(best_model_path, copy_to)
    
    for p in exp_dir_path.glob("*.pth"):
        p.unlink()
        
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


device = torch.device(CFG.device)


all_oof = []
all_true = []
valid_loaders = []
for i, (train_index, valid_index) in enumerate(gkf.split(train, train.target, train.patient_id)):  
    print('#'*25)
    print(f'### Fold {i+1}')
    valid_data = train.iloc[valid_index]

    valid_ds = EEGDataset(valid_data, mode='valid', specs=spectrograms, eeg_specs=all_eegs, raw_eegs=all_raw_eegs, data_type=CFG.data_type)
    valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.batch_size*2, num_workers=3)
    print(f'###  Valid size: {len(valid_index)}')
    print('#'*25)
    
    # # get model
    model_path = f"./best_model_fold{i}.pth"
    if CFG.data_type in ['K','E','KE']:
        print('-----------SpecModel----------------')
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=True, num_classes=6, in_channels=3)
        
    elif CFG.data_type in ['KR','ER','KER']:
        pass
    elif CFG.data_type in ['R']:
        model = WaveNetModel()
        print('-------------WaveNet---------------')
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # # inference
    val_pred = run_inference_loop(model, valid_loader, device)
    valid_loaders.append(valid_loader)
    all_true.append(train.iloc[valid_index][TARGETS].values)
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