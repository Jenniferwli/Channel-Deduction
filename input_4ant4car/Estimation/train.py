import os
gpu_list = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
import numpy as np
import torch
import torch.nn as nn
import random
from model import *

cpu_num = 4 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
torch.set_num_threads(cpu_num )

np.random.seed(1)
random.seed(1)

batch_size=500
epochs=100000
learning_rate=1e-4
print_freq=10   # 每隔多少个epoch打印一次

# 输入给模型的低维部分信道尺寸
input_ant_size=4
input_car_size=4
# 模型要恢复的目标高维完整信道尺寸
ant_size=32
car_size=32

# 根据输入和目标尺寸计算下采样的步长
step_ant=ant_size//input_ant_size
step_car=car_size//input_car_size

depth=8     # Channel Estimation网络深度


model=CDNet(input_ant_size,input_car_size,ant_size,car_size,depth)

if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print('model parameters:', sum(param.numel() for param in model.parameters()))

path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25'  # 数据集获取路径
train_dataset = DatasetFolder(path+'/train')    # 训练集
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset_mobile = DatasetFolder(path+'/specifictest_mobile')    # 移动测试集
test_loader_mobile = torch.utils.data.DataLoader(
    test_dataset_mobile, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

test_dataset_static = DatasetFolder(path+'/specifictest_static')    # 静态测试集
test_loader_static = torch.utils.data.DataLoader(
    test_dataset_static, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)

class CustomSchedule():
    '''
    自定义学习率调度器
    '''
    def __init__(self,d_model,warmup_steps=4000,optimizer=None):
        super(CustomSchedule,self).__init__()
        self.d_model=torch.tensor(d_model,dtype=torch.float32)
        self.warmup_steps=warmup_steps
        self.steps=1.
        self.optimizer=optimizer

    def step(self):
        arg1=self.steps**-0.5
        arg2=self.steps*(self.warmup_steps**-1.5)
        self.steps+=1
        lr=(self.d_model**-0.5)*min(arg1,arg2)
        # cur_lr=min(8e-3,lr)
        cur_lr=lr
        for p in self.optimizer.param_groups:
            p['lr']=cur_lr
        return cur_lr

lr_scheduler=CustomSchedule(d_model=512,warmup_steps=4000,optimizer=optimizer)

nmse_list_mobile=[]
nmse_list_static=[]
stride=6    # 每块边长
choice_in_box=32    # 每个input的序列长度
whole_index=[a for a in range(choice_in_box)]
for epoch in range(epochs):
    for i, input in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        input = input.cuda().float()
        input = input * 10000       # [b,len=32,ant=32,car=32,2]
        index=random.choices(whole_index,k=random.choice(whole_index)+1)   # 从一个batch的32个样本中，随机选择一个子集，子集的长度也是随机的
        index_now=index[-1]
        h_now = input[:, index_now, ...]
        h_now_input = h_now[:, 0:step_ant*(input_ant_size-1)+1:step_ant, 0:step_car*(input_car_size-1)+1:step_car, :]   # 只取部分信道作为模型输入

        h_now_output = model(h_now_input)
        MSE_loss = nn.MSELoss()(h_now_output, h_now)
        loss = MSE_loss
        loss.backward()
        lr_scheduler.step()
        optimizer.step()

        # 每隔20个epoch打印一次
        if epoch %20==0:
            if i%print_freq==0:
                print('lr:%.4e' % optimizer.param_groups[0]['lr'])
                print("epoch :", epoch)
                print("MSELoss : ", loss.item())
                nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
                print("nmse : ", nmse)

    if epoch % 20== 0:
        torch.save(model.state_dict(), './model' + '.pth') # 每20个epoch保存一次模型权重

        #mobile case
        model.eval()
        sum_nmse=0
        for i,input in enumerate(test_loader_mobile):
            input = input.cuda().float()
            input = input * 10000
            index=[a for a in range(input.shape[1])]    # 每个input的序列长度为17
            index_now = index[-1]
            h_now = input[:, index_now, ...]
            h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                          0:step_car * (input_car_size - 1) + 1:step_car, :]
            with torch.no_grad():
                h_now_output = model(h_now_input)
            nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
            sum_nmse += nmse
        avg_nmse=sum_nmse/(i+1)
        print("mobile case: ", avg_nmse)
        nmse_list_mobile.append(avg_nmse)
        nmse_array_mobile = np.array(nmse_list_mobile)
        np.save("nmse_mobile.npy", nmse_array_mobile)

        # static case
        model.eval()
        sum_nmse = 0
        for i, input in enumerate(test_loader_static):
            input = input.cuda().float()
            input = input * 10000
            index = [a for a in range(input.shape[1])]  # 每个input的序列长度为17
            index_now = index[-1]
            h_now = input[:, index_now, ...]
            h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                          0:step_car * (input_car_size - 1) + 1:step_car, :]
            with torch.no_grad():
                h_now_output = model(h_now_input)
            nmse = Nmse(h_now.cpu().detach().numpy(), h_now_output.cpu().detach().numpy())
            sum_nmse += nmse
        avg_nmse = sum_nmse / (i + 1)
        print("static case: ", avg_nmse)
        nmse_list_static.append(avg_nmse)
        nmse_array_static = np.array(nmse_list_static)
        np.save("nmse_static.npy", nmse_array_static)

torch.save(model.state_dict(), './model' + '.pth')