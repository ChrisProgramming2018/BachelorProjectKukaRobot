from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
"""
path = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_1_agent_TD3/events.out.tfevents.1599507860.architect.6470.0"
path1 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_2_agent_TD3/events.out.tfevents.1599507860.architect.6471.0"
path2 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_3_agent_TD3/events.out.tfevents.1599507860.architect.6472.0"
path3 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_4_agent_TD3/events.out.tfevents.1599507860.architect.6473.0"

p1="/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.001_update_freq_50_num_q_target_6_seed_1_agent_TD3_ad/events.out.tfevents.1599731008.architect.25365.0"
p2="/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_2_agent_TD3_ad/events.out.tfevents.1599735305.architect.29334.0"
p3 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_3_agent_TD3_ad/events.out.tfevents.1599735305.architect.29335.0"
p4 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_4_agent_TD3_ad/events.out.tfevents.1599735305.architect.29336.0"
"""

paths = glob.glob("../17.12_pad_0/runs/17.12_pad_0/*")
p = glob.glob("../17.12_pad_8/runs/17.12_pad_8/*")

print("paths", paths)
print("paths", p)

path, path1, path2, path3 = paths
p1, p2, p3, p4 = p

p1 = glob.glob(p1 + "/*")[0]
p2 = glob.glob(p2 + "/*")[0]
p3 = glob.glob(p3 + "/*")[0]
p4 = glob.glob(p4 + "/*")[0]
print("p1", p1)
path = glob.glob(path + "/*")[0]
path1 = glob.glob(path1 + "/*")[0]
path2 = glob.glob(path2 + "/*")[0]
path3 = glob.glob(path3 + "/*")[0]

def createMean(path, path1, path2, path3, name="Reward_mean"):
    value = []
    steps = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == name:
                value.append(v.simple_value)
                steps.append(e.step)
    value1 = []
    steps1 = []
    for e in summary_iterator(path1):
        for v in e.summary.value:
            if v.tag == name:
                value1.append(v.simple_value)
                steps1.append(e.step)
    value2 = []
    steps2 = []
    for e in summary_iterator(path2):
        for v in e.summary.value:
            if v.tag == name:
                value2.append(v.simple_value)
                steps2.append(e.step)
    value3 = []
    steps3 = []
    for e in summary_iterator(path3):

        for v in e.summary.value:
            if v.tag == name:
                value3.append(v.simple_value)
                steps3.append(e.step)
    mean_value = []
    print("len v1 ", len(value))
    print("len v2 ", len(value1))
    print("len v3 ", len(value2))
    print("len v4 ", len(value3))
    for v1, v2, v3, v4 in zip(value, value1, value2, value3):
        mean_value.append((v1+v2+v3+v4)/4.)
    var = []
    for v1, v2, v3, v4, mean in zip(value, value1, value2, value3, mean_value):
        summe = ((v1 -mean)**2 + (v2 -mean)**2 + (v3 -mean)**2 + (v4 -mean)**2)/4.
        var.append(np.sqrt(summe))
    max_mean = []
    min_mean = []
    for v, m in zip(var, mean_value):
        max_mean.append(m+v)
        min_mean.append(m-v)
    return mean_value, min_mean, max_mean


print("Input path ", path)
print("Input path1 ", path1)
print("Input path2 ", path2)
print("Input path3 ", path3)
mean_value, min_mean, max_mean =createMean(path,path1,path2,path3, name= "Reward mean ")

mean_value1, min_mean1, max_mean1 =createMean(p1,p2,p3,p4, name="Reward mean ")

different = []
for e in summary_iterator(path):
    for v in e.summary.value:
        n = v.tag
        if n not in different:
            different.append(n)

print("inside ", different)

different = []
for e in summary_iterator(p1):
    for v in e.summary.value:
        n = v.tag
        if n not in different:
            different.append(n)
print("inside ", different)



print("lenth 1 {} vs 2 {}".format(len(mean_value), len(mean_value1)))

steps = [x for x in range(len(mean_value))]

#plt.plot(steps, min_mean)
#plt.plot(steps, max_mean)
plt.fill_between(steps, min_mean1, max_mean1, alpha=0.2)
l1=plt.plot(steps, mean_value1,color='r', label="TQC_with_Augmentation")
plt.fill_between(steps, min_mean, max_mean, alpha=0.2)
l2=plt.plot(steps, mean_value,color='b', label="TQC")
#plt.legend(handles=[l1, l2], loc='lower right')
plt.legend()
plt.xlabel('1000 steps')
plt.ylabel('Mean Reward')
plt.savefig("TQC_augmentvsTQC.png")
