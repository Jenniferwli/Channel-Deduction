import os
gpu_list = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import torch
import random
import hdf5storage
from model import *
from einops import rearrange

cpu_num = 4 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )

np.random.seed(1)
random.seed(1)

batch_size=1

# 输入给模型的低维部分信道尺寸
input_ant_size=4
input_car_size=4
# 模型要恢复的目标高维完整信道尺寸
ant_size=32
car_size=32

# 根据输入和目标尺寸计算下采样的步长
step_ant=ant_size//input_ant_size
step_car=car_size//input_car_size

depth_est=3     # CMixer网络深度
depth_pred=6    # Transformer信道预测网络深度

model=CDNet(input_ant_size,input_car_size,ant_size,car_size,depth_est,depth_pred)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
model.load_state_dict(torch.load('./model' +'.pth'),strict=False)   # 加载预训练的模型权重

print('model parameters:', sum(param.numel() for param in model.parameters()))

path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat'  # 数据集获取路径
x_trace=np.load('x_trace_smooth.npy')
y_trace=np.load('y_trace_smooth.npy')
len_trace=x_trace.shape[0]



def channel_index(x_index,y_index,path,high=181): 
    '''
    根据坐标返回从mat文件返回信道数据。
    注意：这里的x,y是plot的x轴和y轴，与deepmimo的x轴和y轴正好转置
    '''
    index=x_index*high+y_index
    channel_data_mat=hdf5storage.loadmat(path + '/DeepMIMO_dataset_' + str(index) + '.mat')
    ans=channel_data_mat['DeepMIMO_dataset']
    return ans


truth_channel_list=[]
for i in range(len_trace):
    # 根据坐标获得真实信道数据
    reverse_x_index=x_trace[i]
    y_index=y_trace[i]
    x_index=899-reverse_x_index
    channel_data=channel_index(x_index,y_index,path)

    channel_data=torch.tensor(channel_data) # 将numpy数组转换为PyTorch张量
    channel_data=rearrange(channel_data,'(b ant) car c -> b ant car c',b=1) # 重塑张量维度，增加一个batch维度 (b=1)
    channel_data=channel_data.cuda().float()
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
        # --- 模式1: Working under error propagation ---
        h_now=truth_channel_list[i]
        h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                      0:step_car * (input_car_size - 1) + 1:step_car, :]    # 只取部分信道作为模型输入
        h_past_list=deduction_channel_list[i-past_len:i]    # 用deduction_channel_list中8个过去的完整信道和当前的部分信道预测当前完整信道数据
        h_past_input = torch.stack(h_past_list,dim=0)       # 把h_past_list中的8个元素在第0个维度整合成一个张量
        # print(h_past_input.shape)
        h_past_input=rearrange(h_past_list,'len b ant car c -> b len ant car c')

        with torch.no_grad():
            h_now_output = model(h_now_input, h_past_input)
        nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
        rou = Corr(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy(), ant_size, car_size)
        # 将本次的预测结果 h_now_output 添加到递推列表中，供下一步使用
        deduction_channel_list.append(h_now_output)
        nmse_list.append(nmse)
        rou_list.append(rou)
        if ((i-past_len)%100==0):
            print("error propagation")
            print(nmse,rou)


        # --- 模式2: Working under the ideal case of no error propagation ---
        h_now = truth_channel_list[i]
        h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                      0:step_car * (input_car_size - 1) + 1:step_car, :]    # 只取部分信道作为模型输入
        h_past_list = truth_channel_list[i - past_len:i]    # 用truth_channel_list中8个过去的完整信道和当前的部分信道预测当前完整信道数据
        h_past_input = torch.stack(h_past_list, dim=0)      # 把h_past_list中的8个元素在第0个维度整合成一个张量
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
    "font.family":'Times New Roman',  # 设置字体类型
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