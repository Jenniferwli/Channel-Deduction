import numpy as np
import random
from tqdm import tqdm
import os
gpu_list = '1'
path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25'

np.random.seed(1)
random.seed(1)
ant_size=32
car_size=32

os.makedirs(path+'/specifictest_mobile',exist_ok=True)
os.makedirs(path+'/specifictest_static',exist_ok=True)

stride=6
choice_in_box=32 #一个box内采样的散点个数
test_box=1000
box_choice=20 #每个box内产生20个序列，长度均为17
list_len=17

# mobile case
for i in tqdm(range(test_box)):

    data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
    whole_list = [a for a in range(choice_in_box)]
    for j in range(box_choice):
        list_sample = random.sample(whole_list,list_len)
        new_data=data[list_sample]
        np.save(path  + '/specifictest_mobile' + '/data_' + str(i*box_choice+j) + '.npy',new_data)
np.save(path + '/specifictest_mobile' + '/data_len' + '.npy', test_box * box_choice)

# static case
for i in tqdm(range(test_box)):
    data = np.load(path + '/test' + '/data_' + str(i) + '.npy')
    whole_list = [a for a in range(choice_in_box)]
    for j in range(box_choice):
        list_sample = random.sample(whole_list,list_len-4)
        past_sample=list_sample[0:-1]
        present_sample=list_sample[-1]
        # 将 "当前" 的索引 'present_sample' 添加到 "过去" 的列表中 4 次
        past_sample.append(present_sample)
        past_sample.append(present_sample)
        past_sample.append(present_sample)
        past_sample.append(present_sample)
        
        random.shuffle(past_sample)
        past_sample.append(present_sample)
        list_sample=past_sample
        new_data=data[list_sample]
        np.save(path  + '/specifictest_static' + '/data_' + str(i*box_choice+j) + '.npy',new_data)
np.save(path + '/specifictest_static' + '/data_len' + '.npy', test_box * box_choice)


