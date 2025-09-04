import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from einops import rearrange,repeat


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
        # Space Mixing (Handle the antenna dimension)
        self.ant_mix = nn.Sequential(
            nn.LayerNorm(ant_size*2),   # *2 is because the complex data is represented as two channels: the real part and the imaginary part
            FeedForward(ant_size*2, ant_size*2*2, dropout),
        )
        # Frequency Mixing (Handle the subcarrier dimension)
        self.car_mix = nn.Sequential(
            nn.LayerNorm(car_size*2),
            FeedForward(car_size*2, car_size*2*2, dropout),
        )
    def forward(self, x):   # input、output tensor shape[b ant car channel=2]
        x=rearrange(x,'b ant car c -> b car (ant c)')   # The MLP is applied independently to all antennas for each subcarrier
        x=x+self.ant_mix(x) # Residual connection
        x=rearrange(x,'b car (ant c) -> b ant car c',c=2)

        x=rearrange(x,'b ant car c -> b ant (car c)')   # The MLP is applied independently to all subcarriers for each antenna
        x=x+self.car_mix(x) # Residual connection
        x=rearrange(x,'b ant (car c) -> b ant car c',c=2)

        return x

class MLPMixer_us_2D(nn.Module):
    """
    Multiple MixerBlock_2D modules are stacked to form a deep network.
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
    This is a complete channel estimation network using the CMixer architecture.
    It takes a partial (or low-dimensional) channel as input and reconstructs the full (or high-dimensional) channel.
    """
    def __init__(self,input_ant_size,input_car_size,ant_size,car_size,depth):
        super(est_net, self).__init__()
        self.fc1=nn.Linear(input_ant_size*2,ant_size*2)
        self.fc2=nn.Linear(input_car_size*2,car_size*2)
        self.mlpmixer_us=MLPMixer_us_2D(ant_size,car_size,depth)
        self.fc_reverse1=nn.Linear(ant_size*2,ant_size*2)
        self.fc_reverse2=nn.Linear(car_size*2,car_size*2)

    def forward(self,x): 
        # 1.nput Mapping / Projection: The input antenna and subcarrier dimensions are expanded to the hidden dimensions for the model's internal processing.
        out=rearrange(x,'b in_ant in_car c -> b in_car (in_ant c)')
        out=self.fc1(out)
        out=rearrange(out,'b in_car (ant c) -> b ant (in_car c)',c=2)
        out=self.fc2(out)
        out=rearrange(out,'b ant (car c) -> b ant car c',c=2)

        # 2.Core Mixing: The MLPMixer_us_2D network is used to interleave the spatial and frequency dimensions.
        out=self.mlpmixer_us(out)

        # 3.Output Mapping / Projection: The processed features are mapped back to the final channel dimensions.
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

        # Create two learnable vectors, initialized to zero, for positional encoding to represent 'past' and 'present' sequence elements, respectively
        self.past_embedding = nn.parameter.Parameter(torch.FloatTensor(self.hidden_size), requires_grad=True)
        self.now_embedding = nn.parameter.Parameter(torch.FloatTensor(self.hidden_size), requires_grad=True)
        self.past_embedding.data.fill_(0)
        self.now_embedding.data.fill_(0)

        # 1.Input Linear Layer: Projects the channel data to the Transformer's hidden dimension
        self.fc1=nn.Linear(ant_size*car_size*2,self.hidden_size)
        # 2.Transformer encoder layer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=8, dim_feedforward=self.hidden_size,
            dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformerencoder=torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth_pred)
        # 3.Output Linear Layer: Projects the Transformer's output from the hidden dimension back to the channel data dimension
        self.fc_reverse1=nn.Linear(self.hidden_size,ant_size*car_size*2)

    def forward(self,x_now,x_past):
        # tensor shape x_now:[b ant car c]，x_past[b len-1 ant car c]，out：[b ant car c]
        x_now = rearrange(x_now, 'b ant car c -> b 1 ant car c')
        x_total = torch.cat((x_past, x_now), dim=1)  # place x_now at the end
        x_total = rearrange(x_total, 'b len ant car c -> b len (ant car c)')
        x_total = self.fc1(x_total)

        # Learnable position embeddings
        past_len = x_total.shape[1] - 1
        past_embedding_matrix = repeat(self.past_embedding, 'h -> past_len h', past_len=past_len)
        now_embedding_matrix = rearrange(self.now_embedding, 'h -> 1 h')
        embedding_matrix = torch.cat((past_embedding_matrix, now_embedding_matrix), dim=0)
        batch_size = x_total.shape[0]
        embedding_matrix = repeat(embedding_matrix, 'len h -> b len h', b=batch_size)
        x_total = x_total + embedding_matrix

        out=self.transformerencoder(x_total)
        out=out[:,-1,:]
        out=self.fc_reverse1(out)
        out=rearrange(out,'b (ant car c)-> b ant car c',ant=self.ant_size,car=self.car_size,c=2)
        return out


class CDNet(nn.Module):
    def __init__(self,input_ant_size,input_car_size,ant_size,car_size,depth,depth_pred):
        super(CDNet, self).__init__()
        # 1.use CMixer for channel estimation
        self.est_net=est_net(input_ant_size,input_car_size,ant_size,car_size,depth)
        # 2.use Transformer for channel prediction
        self.pred_net=pred_net(ant_size,car_size,depth_pred)
        # 3.use CMixer to further refine and optimize the Transformer's output 
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
    # Calculate the Normalized Mean Squared Error (NMSE)
    B=input.shape[0]    # batch size
    sum_nmse=0
    # Iterate over each sample in the batch
    for i in range(B):
        # calculate NMSE of single sample
        # The formula is: ||Ground Truth - Prediction||² / ||Ground Truth||²
        # np.linalg.norm calculates the L2 norm (Euclidean norm) by default
        nmse=(np.linalg.norm(input[i]-output[i]))**2/(np.linalg.norm(input[i]))**2

        # Accumulate the NMSE for each sample
        sum_nmse+=nmse

     # Return the average (or mean) NMSE for the batch
    return sum_nmse/B

def cos(a,b):   
    '''
    Calculate the cosine similarity between the predicted channel and the ground truth channel, where both are complex-valued.
    a,b are both complex-valued.
    '''
    numerator=np.linalg.norm(a.dot(np.conjugate(b).T)) 
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    cos=numerator/denominator
    return cos

def Corr(a,b,num_ant,num_car):
    '''
    The function first converts the input data into a complex format, 
    then calculates the cosine similarity for each subcarrier, 
    and finally averages the results over the entire batch and all subcarriers.
    '''
    B = a.shape[0]  # batch size
    # Convert the input data into a complex format
    a = a.reshape([B, num_ant, num_car, 2])
    b = b.reshape([B, num_ant, num_car, 2])
    a_real,a_imag=a[:,:,:,0],a[:,:,:,1]
    b_real, b_imag = b[:, :, :, 0], b[:, :, :, 1]
    a_complex=a_real+1j*a_imag
    b_complex=b_real+1j*b_imag

    # Calculate the cosine similarity for each subcarrier across the batch，
    # and then average the results over the entire batch and all subcarriers.
    sum_rou_batch=0
    for i in range(B):
        sum_rou_car=0
        for j in range(num_car):
            rou=cos(a_complex[i,:,j],b_complex[i,:,j])
            sum_rou_car+=rou
        sum_rou_batch+=sum_rou_car
    avg_rou=sum_rou_batch/(B*num_car)
    return avg_rou