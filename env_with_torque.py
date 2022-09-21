import numpy as np
import pyglet
import math as m

import time

#global x=[]
#for i in range(100,200,1):
    #x.append(i)

class ArmEnv(object):
    
    
    viewer = None
    ddt = 0.001 # refresh rate
    action_bound = [-1.,1.]
    goal = {'x': 110., 'y': 86., 'l': 25}
    state_dim = 13-2
    action_dim = 2
    dis= np.array([1,0])
    dt = 0.001
    m1 = 0.1; m2 = 0.1; 
    
    
    g = 9.8 
    def __init__(self):
        
        
        self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32),('t',np.float32)])
                                 
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = 0  # 2 angles information
        self.on_goal = 0
        self.arm_info['t']=[0,0]
        
        self.alpha1=0
        self.alpha2=0
        self.w1=0
        self.w2=0
        
        self.I1 = 0.01
        self.I2 = 0.01
        
        self.l1=0.5
        self.l2=0.5
        self.g=0
        self.aaa=[]
    def step(self, action):
        #print("ddd")
        done = False
        #self.t = self.t+self.dt
        action = np.clip(action, *self.action_bound)
        
        self.arm_info['t'] += action * self.ddtz
        #print(self.arm_info['t'])
        
        
        (t1, t2) = (self.arm_info['t'] )
        (q1,q2)= self.arm_info['r']
        
        #self.alpha1 = -(self.alpha2*(self.I2 + (self.m2*(self.l2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)) + self.l2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1))))/2)- t1-(self.m2*self.w2*(2*((self.l2*self.w1*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) - 2*((self.l2*self.w1*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + self.l2*(self.w1*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)) - self.l2*(self.w1*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2))))/2 + self.g*self.m2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.g*self.l1*self.m1*m.cos(q1))/2)/(self.I1 + self.I2 + (self.m1*((self.l1**2*m.cos(q1)**2)/2 + (self.l1**2*m.sin(q1)**2)/2))/2 + (self.m2*(2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))**2 + 2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))**2))/2)
        
        #print(self.m2)
        
       # self.w1 = self.alpha1*self.dt + self.w1
        #self.arm_info['r'][0] = self.w1*self.dt + q1
        #print(self.arm_info['r'][0])
        
        #print(self.arm_info['r'])
        self.alpha2 = -((self.m2*(2*((self.l2*self.w1*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(self.w1*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2) - 2*((self.l2*self.w1*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(self.w1*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)))/2 - t2 + ((self.I2 + (self.m2*(self.l2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)) + self.l2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1))))/2)*(t1 + (self.m2*self.w2*(2*((self.l2*self.w1*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) - 2*((self.l2*self.w1*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + self.l2*(self.w1*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)) - self.l2*(self.w1*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2))))/2 - self.g*self.m2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) - (self.g*self.l1*self.m1*m.cos(q1))/2))/(self.I1 + self.I2 + (self.m1*((self.l1**2*m.cos(q1)**2)/2 + (self.l1**2*m.sin(q1)**2)/2))/2 + (self.m2*(2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))**2 + 2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))**2))/2) - (self.m2*self.w2*(self.l2*((self.l2*self.w1*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)) - self.l2*((self.l2*self.w1*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)) + self.l2*(self.w1*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)) - self.l2*(self.w1*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2))))/2 + (self.g*self.l2*self.m2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)/(self.I2 - (self.I2 + (self.m2*(self.l2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)) + self.l2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1))))/2)**2/(self.I1 + self.I2 + (self.m1*((self.l1**2*m.cos(q1)**2)/2 + (self.l1**2*m.sin(q1)**2)/2))/2 + (self.m2*(2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))**2 + 2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))**2))/2) + (self.m2*((self.l2**2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1))**2)/2 + (self.l2**2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2))**2)/2))/2)
        

        #print(self.alpha2)
        
        
        self.w2 = self.alpha2*self.dt + self.w2
        
        self.arm_info['r'][1] = self.w2*self.dt + q2;
        
        #print(self.arm_info['r'])
        
        self.alpha1 = -(self.alpha2*(self.I2 + (self.m2*(self.l2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)) + self.l2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1))))/2)- t1-(self.m2*self.w2*(2*((self.l2*self.w1*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) - 2*((self.l2*self.w1*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + self.l2*(self.w1*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.l2*self.w2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2)*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)) - self.l2*(self.w1*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1)) + (self.l2*self.w2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2)*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2))))/2 + self.g*self.m2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1)) + (self.g*self.l1*self.m1*m.cos(q1))/2)/(self.I1 + self.I2 + (self.m1*((self.l1**2*m.cos(q1)**2)/2 + (self.l1**2*m.sin(q1)**2)/2))/2 + (self.m2*(2*((self.l2*(m.cos(q1)*m.cos(q2) - m.sin(q1)*m.sin(q2)))/2 + self.l1*m.cos(q1))**2 + 2*((self.l2*(m.cos(q1)*m.sin(q2) + m.cos(q2)*m.sin(q1)))/2 + self.l1*m.sin(q1))**2))/2)
        self.w1 = self.alpha1*self.dt + self.w1
        self.arm_info['r'][0] = self.w1*self.dt + q1

        #self.arm_info['r']=self.arm_info['r']
        #8self.arm_info['r'] %= np.pi * 2 
        #print(self.arm_info['r'])
        #time.sleep(0.2)
        ##### with disturbance
        #if i>500 and i<601 or i>700 and i<=800:
            
            #action=np.array([action[0]+self.dis[0],action[1]+self.dis[1]])
       
        #####
        
        
       #self.arm_info['r'] %= np.pi * 2    # normalize
      
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r'] 
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)#-(np.square(action).sum())
        #print(r)
        TD = np.sqrt(dist2[0]**2+dist2[1]**2)
        #print(dist1 + dist2)
        # done and rewardif
        #print(TD)
        self.aaa.append(TD)
        #print(self.aaa)

        #if TD>(self.aaa[len(self.aaa)-2])*1.5:
           #r += -5.
           #done = True
        #elif TD>(self.aaa[len(self.aaa)-2])+0.1:
            
            #r+=-1.
        #else:
           #r+=1
           
           
        #if TD>(self.aaa[len(self.aaa)-2]):
            
            #r+=-0.1
        #else:
            #r+=0.05
         

            
        if (self.goal['x'] - self.goal['l'] < finger[0] < self.goal['x'] + self.goal['l']
        ) and (self.goal['y'] - self.goal['l'] < finger[1] < self.goal['y'] + self.goal['l']):
            r += 20.
            #print("touched")
            self.on_goal += 1
            if self.on_goal ==2:
                r+=1000.
                #print("ff")
                done = True
        else:
            self.on_goal = 0
       
        # state
        s = np.concatenate((self.arm_info['r'],[self.w1]+[self.w2],[self.alpha1]+[self.alpha2],a1xy_/200, finger/2 , [1. if self.on_goal else 0.]))
        #self.arm_info['r'],[self.w1]+[self.w2],[self.alpha1]+[self.alpha2],a1xy_/200, finger/200, dist2 , [1. if self.on_goal else 0.
        #[self.w1]+[self.w2],[self.alpha1]+[self.alpha2]
        
        return s, r, done,TD

    def reset(self):
        self.arm_info['t'] = [0,0]
        self.alpha1=0
        self.alpha2=0
        self.w1=0
        self.w2=0
        #self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.arm_info['r'] =[0,0]
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (q1,q2)= self.arm_info['r']
        a1r=q1
        a2r=q2  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((self.arm_info['r'],[self.w1]+[self.w2],[self.alpha1]+[self.alpha2],a1xy_/200, finger/200 , [1. if self.on_goal else 0.]))
        return s

    def render(self):
        
        
        if self.viewer is None:
            
            self.viewer = Viewer(self.arm_info, self.goal)
        
            
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(0, 0, 0, 0)
        
        self.arm_info = arm_info
        self.center_coord = np.array([200, 200])
        #print(self.arm_info)
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        

    def on_draw(self):
        self.clear()
        pyglet.app.exit()
        self.batch.draw()

    def _update_arm(self):
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        #print(self.arm_info['r'])
        
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        #print(a1tr, a2tr)
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
   
        


if __name__ == '__main__':
    env = ArmEnv()
    #start = time.process_time()
    while True:
        global aaa
        env.render()
        #elapsed = (time.process_time() - start)
        #print(elapsed)
        env.step(env.sample_action())
        #elapsed1 = (time.process_time() - elapsed)
        #print("------",elapsed1)