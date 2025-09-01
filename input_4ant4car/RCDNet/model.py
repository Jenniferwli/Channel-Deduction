import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock_2D(nn.Module):
    def __init__(self, ant_size,car_size,dropout = 0.):
        super().__init__()
        # Space Mixing 空间混合（处理天线维度）
        self.ant_mix = nn.Sequential(
            nn.LayerNorm(ant_size*2),   # *2 是因为复数数据被表示为实部和虚部两个通道
            FeedForward(ant_size*2, ant_size*2*2, dropout),
        )
        # Frequency Mixing 频率混合（处理子载波维度）
        self.car_mix = nn.Sequential(
            nn.LayerNorm(car_size*2),
            FeedForward(car_size*2, car_size*2*2, dropout),
        )
    def forward(self, x):   # 输入、输出张量形状[b ant car channel=2]
        x=rearrange(x,'b ant car c -> b car (ant c)')   # MLP可以独立作用于每个子载波上的所有天线
        x=x+self.ant_mix(x) # 残差链接
        x=rearrange(x,'b car (ant c) -> b ant car c',c=2)

        x=rearrange(x,'b ant car c -> b ant (car c)')   # MLP可以独立作用于每个天线上的所有子载波
        x=x+self.car_mix(x) # 残差链接
        x=rearrange(x,'b ant (car c) -> b ant car c',c=2)

        return x

class MLPMixer_us_2D(nn.Module):
    """
    将多个 MixerBlock_2D 模块堆叠起来，形成一个深度网络。
    """
    def __init__(self,ant_size,car_size,depth,dropout = 0.):
        super(MLPMixer_us_2D, self).__init__()
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock_2D(ant_size,car_size))
    def forward(self,x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x


class est_net(nn.Module):
    """
    一个采用CMixer的完整的信道估计网络。
    它接收一个部分（或低维度）的信道作为输入，然后重建出完整（或高维度）的信道。
    """
    def __init__(self,input_ant_size,input_car_size,ant_size,car_size,depth):
        super(est_net, self).__init__()
        self.fc1=nn.Linear(input_ant_size*2,ant_size*2)
        self.fc2=nn.Linear(input_car_size*2,car_size*2)
        self.mlpmixer_us=MLPMixer_us_2D(ant_size,car_size,depth)
        self.fc_reverse1=nn.Linear(ant_size*2,ant_size*2)
        self.fc_reverse2=nn.Linear(car_size*2,car_size*2)

    def forward(self,x): 
        # 1.输入映射/升维：将输入的天线和子载波维度扩展到模型内部处理的目标维度
        out=rearrange(x,'b in_ant in_car c -> b in_car (in_ant c)')
        out=self.fc1(out)
        out=rearrange(out,'b in_car (ant c) -> b ant (in_car c)',c=2)
        out=self.fc2(out)
        out=rearrange(out,'b ant (car c) -> b ant car c',c=2)

        # 2.核心混合：使用MLPMixer_us_2D网络在空间和频率维度上进行深度特征交叠学习
        out=self.mlpmixer_us(out)

        # 3.输出映射：将处理后的特征映射回最终的信道维度
        out=rearrange(out,'b ant car c -> b car (ant c)')
        out=self.fc_reverse1(out)
        out=rearrange(out,'b car (ant c)-> b ant (car c)',c=2)
        out=self.fc_reverse2(out)
        out=rearrange(out,'b ant (car c) -> b ant car c',c=2)

        return out

class pred_net(nn.Module):
    def __init__(self,ant_size,car_size,depth_pred):
        super(pred_net, self).__init__()
        self.ant_size=ant_size
        self.car_size=car_size
        self.hidden_size=512

        # 1.线性输入层：将信道数据映射到LSTM的隐藏维度
        self.fc1=nn.Linear(ant_size*car_size*2,self.hidden_size)
        # 2.LSTM层
        self.lstm_layer=nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=depth_pred,batch_first=True)
        # 3.线性输出层：将LSTM的输出从隐藏维度映射回信道数据维度
        self.fc_reverse1=nn.Linear(self.hidden_size,ant_size*car_size*2)

    def forward(self,x_now,x_past):
        # 张量形状 x_now:[b ant car c]，x_past[b len-1 ant car c]，out：[b ant car c]
        x_now=rearrange(x_now,'b ant car c -> b 1 ant car c')
        x_total=torch.cat((x_past,x_now),dim=1) #把now放在最后
        x_total=rearrange(x_total,'b len ant car c -> b len (ant car c)')
        x_total=self.fc1(x_total)
        out,(h_n,c_n)=self.lstm_layer(x_total)
        out=out[:,-1,:]
        out=self.fc_reverse1(out)
        out=rearrange(out,'b (ant car c)-> b ant car c',ant=self.ant_size,car=self.car_size,c=2)
        return out


class CDNet(nn.Module):
    def __init__(self,input_ant_size,input_car_size,ant_size,car_size,depth,depth_pred):
        super(CDNet, self).__init__()
        # 1.用CMixer信道估计
        self.est_net=est_net(input_ant_size,input_car_size,ant_size,car_size,depth)
        # 2.用LSTM信道预测
        self.pred_net=pred_net(ant_size,car_size,depth_pred)
        # 3.用CMixer对LSTM的输出进一步精炼和优化
        self.est_net2=est_net(ant_size,car_size,ant_size,car_size,depth)

    def forward(self,input_x_now,x_past):
        x_now=self.est_net(input_x_now)
        out=self.pred_net(x_now,x_past)
        out=self.est_net2(out)
        return out

class DatasetFolder(Dataset):
    def __init__(self, path):
        self.path=path
        self.len=int(np.load(self.path+'/data_len.npy'))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data=np.load(self.path+'/data_'+str(index)+'.npy')
        return data

def Nmse(input,output):
    # 计算归一化均方误差
    B=input.shape[0]    # batch size
    sum_nmse=0
    # 遍历batch中的每一个样本
    for i in range(B):
        # 计算单个样本的NMSE
        # 公式为: ||真实值 - 预测值||² / ||真实值||²
        # np.linalg.norm 默认计算L2范数（欧几里得范数）
        nmse=(np.linalg.norm(input[i]-output[i]))**2/(np.linalg.norm(input[i]))**2

        # 累加每个样本的NMSE
        sum_nmse+=nmse

     # 返回一个batch内的平均NMSE
    return sum_nmse/B

def cos(a,b):   
    '''
    计算预测信道和真实信道之间的余弦相关性，a,b都是复值
    '''
    numerator=np.linalg.norm(a.dot(np.conjugate(b).T))  #分子
    denominator=np.linalg.norm(a)*np.linalg.norm(b)     #分母
    cos=numerator/denominator
    return cos

def Corr(a,b,num_ant,num_car):
    '''
    该函数首先将输入数据转换为复数形式，
    然后逐个子载波计算余弦相似度，最后将所有结果在整个批次和所有子载波上进行平均
    '''
    B = a.shape[0]  # batch size
    # 将输入数据转换为复数形式
    a = a.reshape([B, num_ant, num_car, 2])
    b = b.reshape([B, num_ant, num_car, 2])
    a_real,a_imag=a[:,:,:,0],a[:,:,:,1]
    b_real, b_imag = b[:, :, :, 0], b[:, :, :, 1]
    a_complex=a_real+1j*a_imag
    b_complex=b_real+1j*b_imag

    # 逐个batch和子载波计算余弦相似度,在整个batch和所有子载波上进行平均
    sum_rou_batch=0
    for i in range(B):
        sum_rou_car=0
        for j in range(num_car):
            rou=cos(a_complex[i,:,j],b_complex[i,:,j])
            sum_rou_car+=rou
        sum_rou_batch+=sum_rou_car
    avg_rou=sum_rou_batch/(B*num_car)
    return avg_rou