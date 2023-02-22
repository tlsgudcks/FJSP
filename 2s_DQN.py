# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:28:32 2022

@author: parkh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from FJSP_2s_reinforce import *

learning_rate = 0.001  
gamma = 1
buffer_limit = 10000
batch_size = 16

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit);
    def put(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [],[],[],[],[]
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return torch.tensor(s_lst, dtype=torch. float),torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch. float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(48,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon, env):
        out = self.forward(obs)
        out2 = out.detach().numpy()
        act_list = out2
        act = np.argmax(act_list)
        coin = random.random()
        if coin < epsilon:
            act = env.random_action()
            return act
        else:
            act = env.random_action()
            return act
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        out2 = out.detach().numpy()
        act_list = out2
        act = np.argmax(act_list)
        print(act, act_list)
        return act,act_list
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        #q.number_of_time_list[a] += 1    
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max (1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
params = {
    'MUT': 1,  # 변이확률(%)
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE' : 1000,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESSURE' : 3, # 선택연산의 선택압
    'job_seq' : [y for x in range(1,5) for y in range(1,13)],
    'factory_seq' : [1,2]
    # 원하는 파라미터는 여기에 삽입할 것
    }            
def main():
    env = JAYA_FJSP(params)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 1
    q_load = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    pop, result = env.search()
    for n_epi in range(1000):
        epsilon = max(0.01 , 0.08 - 0.02*(n_epi/200))
        env.reset()
        env.assignment = pop[0][0]
        s = env.s
        s = np.array(s)
        done = False
        score = 0.0
        while not done:
            a = q.sample_action(torch.from_numpy(s). float(), epsilon, env)
            s_prime, r, done = env.step(a)
            done_mask =0.0 if done else 1.0
            if done == False:
                s = np.array(s)
                s_prime = np.array(s_prime)
                memory.put((s,a,r,s_prime,done_mask))
                s = s_prime
                score += r
            if done:
                break
        if memory.size()>1000:
            train(q, q_target, memory, optimizer)
            
        if n_epi % print_interval==0 and n_epi!=0:
            #q_target.load_state_dict(q.state_dict())
            env.reset()
            env.assignment = pop[0][0]
            makespan, critical_machine, Flow_time, util = env.get_fittness2(env.assignment, s_prime)
            print("--------------------------------------------------")
            print("flow time: {}, util : {:.3f}, makespan : {}".format(Flow_time, util, makespan))
            print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval,memory.size(),epsilon*100))
            #score=0.0
        if n_epi % q_load ==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
    env.reset()
    s = env.s
    s = np.array(s)
    env.assignment = pop[0][0]
    done = False
    score = 0.0
    while not done:
        a, b = q.select_action(torch.from_numpy(s). float(), epsilon)
        print(a)
        print(b)
        s_prime, r, done = env.step(a)
        print(r)
        s = np.array(s)
        s_prime = np.array(s_prime)
        s = s_prime
        score += r
        if done:
            break
    env.reset()
    env.assignment = pop[0][0]
    makespan, critical_machine, Flow_time, util = env.get_fittness2(env.assignment, s_prime)
    env.reset()
    env.gannt_chart([pop[0][0], s_prime])
    return makespan, critical_machine, Flow_time, util, score
Flow_time, machine_util, util, makespan, score =main()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
print("Score" , score)
      
    