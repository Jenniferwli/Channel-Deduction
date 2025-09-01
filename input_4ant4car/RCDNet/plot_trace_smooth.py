import numpy as np
import matplotlib.pyplot as plt
import random

# 初始化两个列表，用于存储移动指令的x和y分量
# 初始值为(1, 0)，代表第一次移动的基准是“向右移动一个单位”
random_list_x=[1,]
random_list_y=[0,]

# 最终需要的移动指令数量
len_num=1223
step_size=1
step_num=len_num//step_size

np.random.seed(0)
random.seed(5)

# 轨迹的起始点坐标
start_point_x,start_point_y=110,100

while (len(random_list_x)<step_num):
    x=random.choice([0,1])      # 随机选择一个x方向的移动：0（不动）或 1（向右）
    y=random.choice([-1,0,1])   # 随机选择一个y方向的移动：-1（向下）、0（不动）或 1（向上）
    # 确保新指令(x, y)与上一个指令的“变化幅度”小于1.5，保证了移动“模式”不会发生突变，增加了连续性
    if (np.abs(x-random_list_x[-1])+np.abs(y-random_list_y[-1])<1.5):
        # 确保选择的指令不是(0, 0)这个“原地不动”的指令
        if (np.abs(x)+np.abs(y)>0.5):
            random_list_x.append(x)
            random_list_y.append(y)

print(len(random_list_x))
# 重复step_num中的每一步指令step_size次
final_random_list_x=[]
final_random_list_y=[]
for i in range(step_num):
    for _ in range(step_size):
        final_random_list_x.append(random_list_x[i])
        final_random_list_y.append(random_list_y[i])

x_trace=[start_point_x,]
y_trace=[start_point_y,]
j=0         # 用来追踪当前用到了第几个移动指令
past_len=8
for i in range(past_len+2024):

    # 做一个随机选择，决定是“移动”还是“不动”,两者的概率均为50%
    if (np.random.choice([0, 1], p=np.array([0.5, 0.5])) == 0):
        # 概率为50%，选择“不动”
        # 将上一步的坐标直接加0后添加到轨迹中
        x_trace.append(x_trace[i] + 0)
        y_trace.append(y_trace[i] + 0)
    else:
        # 概率为50%，选择“移动”
        # 将上一步的坐标加上当前指令，得到新坐标
        x_trace.append(x_trace[i]+final_random_list_x[j])
        y_trace.append(y_trace[i]+final_random_list_y[j])
        j=j+1

np.save('x_trace_smooth.npy',np.array(x_trace))
np.save('y_trace_smooth.npy',np.array(y_trace))


from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 12,
}
rcParams.update(config)
palette = plt.get_cmap('Set1')
plt.figure(figsize=(899/50,180/50))
plt.ylim(0,180)
plt.xlim(0,899)
# 绘制第一部分：初始轨迹（已知信道）
plt.plot(x_trace[0:past_len],y_trace[0:past_len],'-',color='green',label='Known channels at initial trasmission')
# 绘制第二部分：后续轨迹（需获取信道）
plt.plot(x_trace[past_len:-1],y_trace[past_len:-1],'--',marker='o',markersize=4,markevery=100,color='orange',label='Channels required acquisition')
plt.xticks([])
plt.yticks([])
plt.legend(prop={'size':15},markerscale=1.5)
plt.savefig('trace_smooth.jpg')
plt.savefig('trace_smooth.eps')
plt.savefig('trace_smooth.pdf')
plt.show()