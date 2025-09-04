import numpy as np
import hdf5storage
import random
from tqdm import tqdm
import os
gpu_list = '1'
path='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25'
path_mat='/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat'

np.random.seed(1)
random.seed(1)
ant_size=32
car_size=32

os.makedirs(path+'/train',exist_ok=True)
os.makedirs(path+'/test',exist_ok=True)

x_length=181
y_length=900

stride=6    # Side length of each block

# The scene is divided into blocks
x_box=x_length//stride
y_box=y_length//stride
total_box=x_box*y_box
print("total box:",total_box)

train_box=3000
test_box=1000
choice_in_box=32
box_index=np.arange(total_box)
np.random.shuffle(box_index)

for i in tqdm(range(train_box)):
    # First, infer the grid coordinates (row and column index) of the i-th box.
    x_box_index=box_index[i]%x_box
    y_box_index=box_index[i]//x_box

    pianyi_list = [a for a in range(stride * stride)]
    pianyi_list_sample = random.sample(pianyi_list,choice_in_box)

    # zuobiao_list = []
    data_list = []
    for j in pianyi_list_sample:
        # Then, infer the relative horizontal and vertical coordinates of the j-th data point within the box
        x_pianyi = j % stride
        y_pianyi = j // stride
        basic_zuobiao = y_box_index * x_length * stride + x_box_index * stride
        zuobiao = basic_zuobiao + y_pianyi * x_length + x_pianyi
        # zuobiao_list.append(zuobiao)
        data_mat = hdf5storage.loadmat(path_mat + '/DeepMIMO_dataset_' + str(zuobiao) + '.mat')
        data = data_mat['DeepMIMO_dataset']
        data_list.append(data)
    random.shuffle(data_list)
    data_numpy = np.array(data_list)
    np.save(path  + '/train' + '/data_' + str(i) + '.npy', data_numpy)
np.save(path+'/train'+'/data_len'+'.npy',train_box)

for i in tqdm(range(test_box)):
    # First, infer the grid coordinates (row and column index) of the i-th box.
    x_box_index=box_index[i+train_box]%x_box
    y_box_index=box_index[i+train_box]//x_box

    pianyi_list = [a for a in range(stride * stride)]
    pianyi_list_sample = random.sample(pianyi_list,choice_in_box)

    # zuobiao_list = []
    data_list = []
    for j in pianyi_list_sample:
        # Then, infer the relative horizontal and vertical coordinates of the j-th data point within the box
        x_pianyi = j % stride
        y_pianyi = j // stride
        basic_zuobiao = y_box_index * x_length * stride + x_box_index * stride
        zuobiao = basic_zuobiao + y_pianyi * x_length + x_pianyi
        # zuobiao_list.append(zuobiao)
        data_mat = hdf5storage.loadmat(path_mat + '/DeepMIMO_dataset_' + str(zuobiao) + '.mat')
        data = data_mat['DeepMIMO_dataset']
        data_list.append(data)
    random.shuffle(data_list)
    data_numpy = np.array(data_list)
    np.save(path  + '/test' + '/data_' + str(i) + '.npy', data_numpy)
np.save(path+'/test'+'/data_len'+'.npy',test_box)
