"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""

from env_with_torque_1 import ArmEnv
#from env_with_torque import Viewer
from rl import DDPG
import numpy as np
import matplotlib.pyplot as plt
import time


#from xlwt import Workbook
#wb = Workbook()

#sheet1 = wb.add_sheet('reward')

MAX_EPISODES = 700
MAX_EP_STEPS = 1000
ON_TRAIN = True

# set env

# set env
env = ArmEnv()
#env1=Viewer
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
total_rewards_hist=[]
TempD=[]

rl = DDPG(a_dim, s_dim, a_bound)
var = 2.
steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        
        #wb.save('reward.xls')
        #start = time.process_time()
        for j in range(MAX_EP_STEPS):
            
            if i>5:
                env.render()

            a = rl.choose_action(s)
            #print(a)
            
            #a = np.clip(np.random.normal(a, var), *a_bound)
            
            s_, r, done,TD = env.step(a)
            
            
            rl.store_transition(s, a, r, s_)
            #sheet1.write(j, i, r)
            
            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            
            if done or j == MAX_EP_STEPS-1:
                #print(a)
                total_rewards_hist.append(ep_r)
                TempD.append(TD)
                
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
        #elapsed = (time.process_time() - start)
        #print(elapsed)
        #time.sleep(5)
    rl.save()
    #if Viewer is not None:
    #figure, axis = plt.subplots(1, 2)
    #t = np.arange(50)
    #axis[0, 0].plot(t, total_rewards_hist)
    #axis[0, 0].set_title("Reward VS Episode")
   
   # axis[0, 0].set_xlabel("Episode")
    #axis[0, 0].set_ylabel("Reward")

   # axis[0, 1].plot(t, TempD)
   # axis[0, 1].set_title(" temporal difference  vs Episode")
   # axis[0, 1].set_xlabel(" temporal difference")
   # axis[0, 1].set_ylabel("Epsiode")
  #  ax.legend()
   # plt.show()  
    fig, ax = plt.subplots()
    t = np.arange(700)
    ax.plot(t, TempD)
    
    ax.set_title("temporal difference  vs Episode")
    ax.set_xlabel(" Episode")
    ax.set_ylabel("temporal difference ")
    
    plt.show()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(1200):
            env.render()
            a = rl.choose_action(s)
            s, r, done,TD = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()



