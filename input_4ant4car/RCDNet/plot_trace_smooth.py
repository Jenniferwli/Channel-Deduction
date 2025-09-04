import numpy as np
import matplotlib.pyplot as plt
import random

# Initialize two lists to store the x and y components of the movement instructions
# The initial value is (1, 0), representing the baseline for the first move as 'move right by one unit'.
random_list_x=[1,]
random_list_y=[0,]

# The total number of required movement instructions
len_num=1223
step_size=1
step_num=len_num//step_size

np.random.seed(0)
random.seed(5)

# The starting point coordinates of the trajectory
start_point_x,start_point_y=110,100

while (len(random_list_x)<step_num):
    x=random.choice([0,1])      # Randomly select a move in the x-direction: 0 (no move) or 1 (move right)
    y=random.choice([-1,0,1])   # Randomly select a move in the y-direction: -1 (down), 0 (no move), or 1 (up)
    # Ensure the magnitude of change (Euclidean distance) between the new instruction (x, y) and the previous one is less than 1.5. 
    # This prevents abrupt changes in the movement pattern and adds continuity
    if (np.abs(x-random_list_x[-1])+np.abs(y-random_list_y[-1])<1.5):
        # Ensure the selected instruction is not the 'no move' command (0, 0)
        if (np.abs(x)+np.abs(y)>0.5):
            random_list_x.append(x)
            random_list_y.append(y)

print(len(random_list_x))
# Repeat each instruction in step_num for step_size times
final_random_list_x=[]
final_random_list_y=[]
for i in range(step_num):
    for _ in range(step_size):
        final_random_list_x.append(random_list_x[i])
        final_random_list_y.append(random_list_y[i])

x_trace=[start_point_x,]
y_trace=[start_point_y,]
j=0         # Used to track the index of the current movement instruction
past_len=8
for i in range(past_len+2024):

    #Make a random choice to either 'move' or 'stay', each with a 50% probability
    if (np.random.choice([0, 1], p=np.array([0.5, 0.5])) == 0):
        # With a 50% probability, choose to 'stay'
        # Append the previous coordinates to the trajectory without change (effectively adding (0,0))
        x_trace.append(x_trace[i] + 0)
        y_trace.append(y_trace[i] + 0)
    else:
        # With a 50% probability, choose to 'move'
        # Add the current instruction to the previous coordinates to get the new coordinates
        x_trace.append(x_trace[i]+final_random_list_x[j])
        y_trace.append(y_trace[i]+final_random_list_y[j])
        j=j+1

np.save('x_trace_smooth.npy',np.array(x_trace))
np.save('y_trace_smooth.npy',np.array(y_trace))


from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # Set the font type
    "font.size": 12,
}
rcParams.update(config)
palette = plt.get_cmap('Set1')
plt.figure(figsize=(899/50,180/50))
plt.ylim(0,180)
plt.xlim(0,899)
# Plot Part 1: Initial Trajectory (Known Channel)
plt.plot(x_trace[0:past_len],y_trace[0:past_len],'-',color='green',label='Known channels at initial trasmission')
# Plot Part 2: Subsequent Trajectory (Channel to be Acquired)
plt.plot(x_trace[past_len:-1],y_trace[past_len:-1],'--',marker='o',markersize=4,markevery=100,color='orange',label='Channels required acquisition')
plt.xticks([])
plt.yticks([])
plt.legend(prop={'size':15},markerscale=1.5)
plt.savefig('trace_smooth.jpg')
plt.savefig('trace_smooth.eps')
plt.savefig('trace_smooth.pdf')
plt.show()