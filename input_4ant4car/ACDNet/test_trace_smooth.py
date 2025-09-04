import os
gpu_list = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import torch
import random
import hdf5storage
from model import *
from einops import rearrange

cpu_num = 4 # Set this to the number of CPU cores you want to use
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )

np.random.seed(1)
random.seed(1)

batch_size=1

# The dimension of the low-dimensional partial channel input to the model
input_ant_size=4
input_car_size=4
# The target dimension of the high-dimensional full channel for the model to reconstruct
ant_size=32
car_size=32

# Calculate the downsampling stride based on the input and target dimensions
step_ant=ant_size//input_ant_size
step_car=car_size//input_car_size

depth_est=3     # Depth of the CMixer network
depth_pred=6    # Depth of the Transformer channel prediction network

model=CDNet(input_ant_size,input_car_size,ant_size,car_size,depth_est,depth_pred)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
model.load_state_dict(torch.load('./model' +'.pth'),strict=False)   # Load the pre-trained model weights

print('model parameters:', sum(param.numel() for param in model.parameters()))

path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat'  # Path to the dataset
x_trace=np.load('x_trace_smooth.npy')
y_trace=np.load('y_trace_smooth.npy')
len_trace=x_trace.shape[0]



def channel_index(x_index,y_index,path,high=181): 
    '''
    Returns channel data from the .mat file based on the given coordinates.
    Note: The x and y here refer to the plot's x-axis and y-axis, which are transposed relative to the x and y axes in the DeepMIMO dataset.
    '''
    index=x_index*high+y_index
    channel_data_mat=hdf5storage.loadmat(path + '/DeepMIMO_dataset_' + str(index) + '.mat')
    ans=channel_data_mat['DeepMIMO_dataset']
    return ans


truth_channel_list=[]
for i in range(len_trace):
    # Get the ground truth channel data based on the coordinates
    reverse_x_index=x_trace[i]
    y_index=y_trace[i]
    x_index=899-reverse_x_index
    channel_data=channel_index(x_index,y_index,path)

    channel_data=torch.tensor(channel_data) # Convert the NumPy array to a PyTorch tensor
    channel_data=rearrange(channel_data,'(b ant) car c -> b ant car c',b=1) # Reshape the tensor to add a batch dimension (b=1)
    channel_data=channel_data*10000
    truth_channel_list.append(channel_data)

deduction_channel_list=[]
nmse_list=[]
rou_list=[]
nmse_list_truth=[]
rou_list_truth=[]
past_len=8
for i in range(len_trace):
    if (i<past_len):
        deduction_channel_list.append(truth_channel_list[i])
    else:
        # --- Mode1: Working under error propagation ---
        h_now=truth_channel_list[i]
        h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                      0:step_car * (input_car_size - 1) + 1:step_car, :]    # Use only a partial channel as the model input
        h_past_list=deduction_channel_list[i-past_len:i]    # Use the 8 past full channel states from deduction_channel_list and the current partial channel to predict the current full channel
        h_past_input = torch.stack(h_past_list,dim=0)       # Stack the 8 elements from h_past_list into a single tensor along dimension 0
        # print(h_past_input.shape)
        h_past_input=rearrange(h_past_list,'len b ant car c -> b len ant car c')

        with torch.no_grad():
            h_now_output = model(h_now_input, h_past_input)
        nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
        rou = Corr(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy(), ant_size, car_size)
        # Append the current prediction result, h_now_output, to the history list for use in the next step
        deduction_channel_list.append(h_now_output)
        nmse_list.append(nmse)
        rou_list.append(rou)
        if ((i-past_len)%100==0):
            print("error propagation")
            print(nmse,rou)


        # --- Mode2: Working under the ideal case of no error propagation ---
        h_now = truth_channel_list[i]
        h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                      0:step_car * (input_car_size - 1) + 1:step_car, :]    # Use only a partial channel as the model input
        h_past_list = truth_channel_list[i - past_len:i]    # Use the 8 past full channel states from truth_channel_list and the current partial channel to predict the current full channel
        h_past_input = torch.stack(h_past_list, dim=0)      # Stack the 8 elements from h_past_list into a single tensor along dimension 0.
        # print(h_past_input.shape)
        h_past_input = rearrange(h_past_list, 'len b ant car c -> b len ant car c')

        with torch.no_grad():
            h_now_output = model(h_now_input, h_past_input)
        nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
        rou = Corr(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy(), ant_size, car_size)
        nmse_list_truth.append(nmse)
        rou_list_truth.append(rou)
        if ((i - past_len) % 100 == 0):
            print("no error propagation")
            print(nmse, rou)

print('error propagation:',sum(nmse_list)/len(nmse_list))
print('no error propagation:',sum(nmse_list_truth)/len(nmse_list_truth))

import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # Set the font type
    "font.size": 26,
}
rcParams.update(config)
palette = plt.get_cmap('Set1')
plt.figure(figsize=(899/50,180/50+1.5))
plt.rcParams['axes.linewidth'] = 2
plt.ylim(0.0001,2)
plt.yscale('log')
plt.ylabel('NMSE')

plt.plot(range(len_trace-past_len)[::5],nmse_list[::5],linewidth=2,color=palette(1),
         linestyle='-',label='ACDNet (Working under error propagation)')
plt.plot(range(len_trace-past_len)[::5],nmse_list_truth[::5],linewidth=2,color=palette(2),
         linestyle='-',label='ACDNet (Working under the ideal case of no error propagation)')
plt.legend()
plt.xlabel('Time slots')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('nmse_trace_ACDNet.jpg')
plt.savefig('nmse_trace_ACDNet.eps')
plt.savefig('nmse_trace_ACDNet.pdf')
plt.show()