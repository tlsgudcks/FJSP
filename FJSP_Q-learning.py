# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:42:32 2022

@author: parkh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

import random
import numpy as np
import random
import copy
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('C:/Users/parkh/FJSP2.csv',index_col=(0)) #job과 operation을 기록한 테이블
df2 = pd.read_csv('C:/Users/parkh/FJSP_SETUP2.csv', index_col=(0)) #setup time 테이블


params = {
    'MUT': 1,  # 변이확률(%)
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE' : 100,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESSURE' : 3, # 선택연산의 선택압
    'p_table' : pd.read_csv('C:/Users/parkh/FJSP2.csv',index_col=(0)),
    's_table' : pd.read_csv('C:/Users/parkh/FJSP_SETUP2.csv', index_col=(0)),
    'machine_seq' : ['M1','M2','M3'], #총 머신 
    'job_seq' : [y for x in range(1,6) for y in range(1,4)]
    # 원하는 파라미터는 여기에 삽입할 것
    }

class FJSP():
    def __init__(self,parameters):
        self.params = {}
        for key, value in parameters.items():
            self.params[key] = value
        self.s= ""
        self.assignment = []
        self.job_endTime = {'j1':0, 'j2':0, 'j3':0}
        self.machine_endTime={'M1':0,'M2':0,'M3':0}
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0"}
        self.preOperation={'j1':1,'j2':1,'j3':1}
        self.k=0
        self.c_max=0
        
    def step(self, a):
        if a==0:
            self.select_M1() 
            reward=self.return_reward()
        elif a==1:
            self.select_M2()
            reward=self.return_reward()
        elif a==2:
            self.select_M3()
            reward=self.return_reward()
        done = self.is_done()
        return self.s, reward, done
    
    def select_M1(self):
        self.s += "1"
    def select_M2(self): 
        self.s += "2"
    def select_M3(self):
        self.s += "3"
    
    def is_done(self):
        if len(self.s) == 15:
            return True
        else:
            return False
        
    def start(self):
        random.shuffle(self.params["job_seq"])
        for op in self.params["job_seq"]:
            job = "j"+str(op)
            op=job+str(self.preOperation[job])
            self.assignment.append(op)
            self.preOperation[job] +=1
            if self.preOperation[job] == 6:
                self.preOperation[job] = 1
        print(self.assignment)
        
        
    def start2(self):
        self.assignment=['j11', 'j31', 'j32', 'j12', 'j33', 
                         'j13', 'j34', 'j14', 'j35', 'j21', 
                         'j22', 'j23', 'j24', 'j15', 'j25']
        print(self.assignment)
        
    def reset(self):
        self.k=0
        self.s=""
        self.job_endTime = {'j1':0, 'j2':0, 'j3':0}
        self.machine_endTime={'M1':0,'M2':0,'M3':0}
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0"}
        self.preOperation={'j1':1,'j2':1,'j3':1}
        self.c_max = 0
        return self.s
    
    def return_reward(self):
        jobOp=self.assignment[len(self.s)-1]         #'j11'의 형태를 j1로 
        job=jobOp[0:2]
        machine="M"+self.s[len(self.s)-1]
        df2_sorted = self.params["s_table"][job] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
        setup_time=df2_sorted.loc[self.machine_prejob[machine]] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        detach_setup_time=0
        if self.job_endTime[job]>self.machine_endTime[machine]:
            remain_time = self.job_endTime[job] - self.machine_endTime[machine]
            if remain_time - setup_time >= 0:
                detach_setup_time = setup_time
            else:
                detach_setup_time = setup_time-remain_time
        time = max(self.machine_endTime[machine] ,self.job_endTime[job]) #machine과 job의 순서 제약조건을 지키기 위해 더 큰 값을 설정함
        time = time - detach_setup_time
        df_sorted = self.params["p_table"][machine] #p_time테이블에서 현재 machine에 해당하는 열을 가져옴
        p_time = df_sorted.loc[jobOp] #해당하는 job과 operation의 시간을 가져옴
        time = time+p_time + setup_time # 프로세스타임과 셋업타임을 더해줌
        self.machine_endTime[machine]=time #기계의 끝나는 시간 설정
        self.job_endTime[job]=time #job의 끝나는 시간 설정
        self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        all_values = self.machine_endTime.values()
        c_max=max(all_values)
        reward = c_max-self.c_max
        reward = 30-reward
        self.c_max = c_max
        return reward
    
class QAgent():
    def __init__(self):
        self.q_table = np.zeros((200000000,3))
        self.eps = 0.9
        self.alpha = 0.001
    
    def get_state(self, s):
        state = 0
        for i in range(len(s)):
            if s[i]=="1":
                state = state + 3**(i)
            elif s[i]=="2":
                state = state + 2 * 3**(i)
            elif s[i]=="3":
                state = state + 3 * 3**(i)
        return state
        
    def select_action(self, s):
        coin = random.random()
        k = self.get_state(s)
        if coin < self.eps:
            action = random.randint(0,2)
        else:
            action_val = self.q_table[k,:]
            action = np.argmax(action_val)
        return action
    
    def select_action2(self, s):
        k = self.get_state(s)
        action_val = self.q_table[k,:]
        action = np.argmax(action_val)
        print(self.q_table[k,:])
        return action
    
    def update_table(self, transition):
        #"",1,1,1,False
        s, a, r, s_prime = transition
        k = self.get_state(s)
        #k=0
        next_k = s_prime
        next_k= self.get_state(next_k)
        #SARSA 업데이트 식을 이용
        self.q_table[k,a] = self.q_table[k,a] + self.alpha * (r + np.amax(self.q_table[next_k, :]) - self.q_table[k,a])
   
    def anneal_eps(self):
        self.eps -=0.0002
        self.eps = max(self.eps, 0.2)
    
    def show(self):
        print(self.q_table.tolist())
        print(self.eps)
        
def main():
    env = FJSP(params)
    agent = QAgent()
    env.start2()
    for n_epi in range(5000):
        done = False
        
        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s=s_prime
        agent.anneal_eps()
    done=False
    s=env.reset()
    r_sum=0
    #agent.show()
    while not done:
        a = agent.select_action2(s)
        s_prime, r, done = env.step(a)
        r_sum= r_sum+r
        s = s_prime
        print(a)
    #agent.show()
    
    return r_sum,s
    #agent.show_table()
av=0
r_sum_list=[]
for i in range(100):
    r_sum,s=main()
    best_r=450-r_sum
    av += best_r
    r_sum_list.append([best_r,s])
    print(i+1 , "회 최적정책 리워드는 ", best_r)
print(av/100)
print(max(r_sum_list))
print(min(r_sum_list))
