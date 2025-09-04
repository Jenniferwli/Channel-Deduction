import os
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import torch
import random
from model import *

cpu_num = 4 # Set this to the number of CPU cores you want to use
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )

np.random.seed(1)
random.seed(1)

batch_size=500

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
depth_pred=2    # Depth of the LSTM channel prediction network

model=CDNet(input_ant_size,input_car_size,ant_size,car_size,depth_est,depth_pred)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
model.load_state_dict(torch.load('./model' +'.pth'),strict=False)   # Load the pre-trained model weights

print('model parameters:', sum(param.numel() for param in model.parameters()))

path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25'  # Path to the dataset
train_dataset = DatasetFolder(path+'/train')    # Training set
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset_mobile = DatasetFolder(path+'/specifictest_mobile')    # Mobile test set
test_loader_mobile = torch.utils.data.DataLoader(
    test_dataset_mobile, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset_static = DatasetFolder(path+'/specifictest_static')    # Static test set
test_loader_static = torch.utils.data.DataLoader(
    test_dataset_static, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

#mobile case
model.eval()
sum_nmse=0
sum_rou=0
for i,input in enumerate(test_loader_mobile):
    input = input.cuda().float()
    input = input * 10000
    index=[a for a in range(input.shape[1])]    # The sequence length for each input is 17
    index_now = index[-1]
    index_past = index[0:-1]
    h_now = input[:, index_now, ...]
    h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                  0:step_car * (input_car_size - 1) + 1:step_car, :]    # Use only a partial channel as the model input
    h_past_input = input[:, index_past, ...]    # The channels for the "past" sequence are complete
    with torch.no_grad():
        h_now_output = model(h_now_input, h_past_input)
    nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
    rou = Corr(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy(), ant_size, car_size)
    sum_nmse += nmse
    sum_rou += rou
avg_nmse = sum_nmse / (i + 1)
avg_rou = sum_rou / (i + 1)
print("mobile case: ", avg_nmse)
print("mobile case: ", avg_rou)


#static case
model.eval()
sum_nmse=0
sum_rou=0
for i,input in enumerate(test_loader_static):
    input = input.cuda().float()
    input = input * 10000
    index=[a for a in range(input.shape[1])]    # The sequence length for each input is 17
    index_now = index[-1]
    index_past = index[0:-1]
    h_now = input[:, index_now, ...]
    h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                  0:step_car * (input_car_size - 1) + 1:step_car, :]    # Use only a partial channel as the model input
    h_past_input = input[:, index_past, ...]    # The channels for the "past" sequence are complete
    with torch.no_grad():
        h_now_output = model(h_now_input, h_past_input)
    nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
    rou = Corr(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy(), ant_size, car_size)
    sum_nmse += nmse
    sum_rou += rou
avg_nmse = sum_nmse / (i + 1)
avg_rou = sum_rou / (i + 1)
print("static case: ", avg_nmse)
print("static case: ", avg_rou)