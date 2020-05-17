import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import copy
import time
import matplotlib.pyplot as plt

class Object():
    '''Object in the game, food or player'''
    def __init__(self,xrange=(0,10),yrange=(0,10)):
        self.x=random.randint(*xrange)
        self.y=random.randint(*yrange)

    def __repr__(self):
        return "({},{})".format(self.x,self.y)
        
class Game():
    '''Game object that holds the state of the board'''
    def __init__(self,xrange=(0,10),yrange=(0,10)):
        self.xrange=xrange
        self.yrange=yrange
    
    def reset(self):
        self.food=Object(self.xrange,self.yrange)
        self.player=Object(self.xrange,self.yrange)
        return self

    def step(self,action):
        old_state=copy.deepcopy(self) #to avoid changes to player and food attrs
        if action==0:
            self.player.y+=1
        elif action==1:
            self.player.x+=1
        elif action==2:
            self.player.y-=1
        elif action==3:
            self.player.x-=1
        elif action==4:
            pass
        else:
            raise ValueError("Action not accepted, tried {}".format(action))
        new_state=copy.deepcopy(self)
        reward=self.get_reward(old_state,new_state)
        return self,reward

    def get_reward(self,old_state,new_state):
        old_d=np.sqrt((old_state.player.x-old_state.food.x)**2+(old_state.player.y-old_state.food.y)**2)
        new_d=np.sqrt((new_state.player.x-new_state.food.x)**2+(new_state.player.y-new_state.food.y)**2)
        if new_d==0:
            self.reset()
            return 300
        elif old_d<=new_d: #if do nothing and not on food take points away
            return -15/new_d
        elif new_d<old_d:
            return 10/new_d

    def __repr__(self):
        return repr(self.food)+repr(self.player)

    def as_tensor(self):
        return tf.convert_to_tensor([[self.food.x,self.food.y,self.player.x,self.player.y]],dtype=tf.float32)

    def state(self):
        return ((self.food.x,self.food.y),(self.player.x,self.player.y))

def take_step(env,obs,model,loss_fn):
    '''Take a step and compute the new board, reward, and gradient'''
    with tf.GradientTape() as tape:
        pred=model(obs.as_tensor())
        summed=[sum(pred[0][:i].numpy()) for i in range(pred.shape[1]+1)]
        n=random.random()
        action= np.argmax([a>n for a in summed])-1
        y_target = tf.cast(action,tf.float32)
        loss=loss_fn(y_target,pred)
    grad=tape.gradient(loss,model.trainable_variables)
    obs,reward = env.step(action)
    return obs,reward,grad



def play_multiple(env,episodes,max_steps,model,loss_fn):
    rewards=[]
    grads=[]
    for episode in range(episodes):
        current_rewards=[]
        current_grads=[]
        obs=env.reset()
        for step in range(max_steps):
            obs,reward,grad = take_step(env,obs,model,loss_fn)
            current_rewards.append(reward)
            current_grads.append(grad)
        rewards.append(current_rewards)
        grads.append(current_grads)
    return rewards,grads

def discount_rewards(rewards,discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2,-1,-1):
        discounted[step]+=discounted[step+1]*discount_factor
    return discounted

def discount_and_norm(all_rewards,discount_factor):
    all_discount_rewards=[discount_rewards(reward,discount_factor) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discount_rewards)
    reward_mean=np.mean(flat_rewards)
    reward_std=np.std(flat_rewards)
    return [(discount_rewards - reward_mean)/reward_std for discount_rewards in all_discount_rewards]

def scorer(rewards):
    '''Simple scorer to calculate percent correct moves and food eaten'''
    i=0
    n=0
    c=0
    for reward in rewards:
        if reward==300:
            i+=1
            n+=1
            c+=1
        elif reward>0:
            i+=1
            n+=1
        elif reward<0:
            n+=1
    return i/n,c

class Solver():
    def __init__(self):
        self.env=Game(xrange=(0,20),yrange=(0,20))

    def build_network(self):
        #in: food x, food y, player x, player y
        #out: 0: up, 1: right, 2: down, 3:left
        self.model=keras.models.Sequential([
            keras.layers.Dense(25,activation='elu',input_shape=[4],name='layer1'),
            keras.layers.Dense(25,activation='elu',name='layer2'),
            keras.layers.Dense(4,activation='softmax',name='layer3'),
            ])
        
    def train(self,n_iter,n_episodes_per_update,n_max_steps,discount_factor,plot=False,verbose=True):
        self.build_network()
        optimizer=keras.optimizers.Adam(lr=0.01)
        loss_fn=keras.losses.SparseCategoricalCrossentropy()
        self.every_reward=[]
        start=time.time()    
        for iteration in range(n_iter):
            all_rewards,all_grads = play_multiple(self.env,n_episodes_per_update,n_max_steps,self.model,loss_fn)
            self.every_reward.append(all_rewards)
            all_final_rewards = discount_and_norm(all_rewards,discount_factor)
            all_mean_grads=[]
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]
                     for episode_index, final_rewards in enumerate(all_final_rewards)
                     for step,final_reward in enumerate(final_rewards)],axis=0)
                all_mean_grads.append(mean_grads)
            optimizer.apply_gradients(zip(all_mean_grads,self.model.trainable_variables))

            if iteration%10==0:
                if verbose==True:
                    score=[scorer(reward) for reward in all_rewards]
                    print("Iteration: {}, time: {}, avg percent correct: {}, avg num eaten: {}".format(iteration,time.time()-start,np.average([s[0] for s in score]),np.average([s[1] for s in score])))       

        self.scores=[np.average([scorer(reward)[1] for reward in rewards]) for rewards in self.every_reward]
        if plot==True:
            plt.plot(np.arange(1,len(self.scores)+1,1),self.scores)
            plt.show()

    def test(self,n_steps,verbose=True):
        obs=self.env.reset()
        result=[]
        for i in range(n_steps):
            pred=self.model(obs.as_tensor())[0]
            summed=[sum(pred[:i].numpy()) for i in range(pred.shape[-1]+1)]
            n=random.random()
            action= np.argmax([a>n for a in summed])-1
            obs,reward=self.env.step(action)
            if verbose==True:
                print(self.env,reward)
            result.append((self.env.state(),reward))
        return result


    def load(self,path):
        self.model = tf.keras.models.load_model(path,compile=False)

    def save(self,path):
        self.model.save(path)
        
clf=Solver()
#clf.train(n_iter=250,n_episodes_per_update=25,n_max_steps=100,discount_factor=0.95) #train
#clf.load('rl_model') #load using training stats above
result=clf.test(n_steps=100)
