# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 01:02:50 2022

@author: parkh
"""

import random
import copy
import pandas as pd
import numpy as np
from datetime import datetime

params = {
    'MUT': 1,  # 변이확률(%)
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터 (%)
    'POP_SIZE' : 100,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'SELECTION_PRESSURE' : 3, # 선택연산의 선택압
    'list_seq' : [1,2,3,4,5,6,7,8,9,10],
    'machine_seq': ['M1','M2','M3','M4'], #총 머신 시퀀스
    'job_seq' : [y for x in range(1,6) for y in range(1,5)]
    # 원하는 파라미터는 여기에 삽입할 것
    }
class JAYA_FJSP():
    def __init__(self, parameters):
        self.eps = 0.9
        self.params = {}
        for key, value in parameters.items():
            self.params[key] = value
        self.p_table = pd.read_csv('C:/Users/parkh/FJSP5.csv',index_col=(0)) #job과 operation을 기록한 테이블
        self.s_table = pd.read_csv('C:/Users/parkh/FJSP_SETUP.csv', index_col=(0)) #setup time 테이블
        self.job_endTime={'j1':0, 'j2':0, 'j3':0, 'j4':0} # job의 끝나는 지점을 등록
        self.machine_endTime={'M1':0,'M2':0,'M3':0,'M4':0} # machine의 끝나는 지점을 등록
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0", 'M4':"j0"}
        self.job_preOperation={'1':1,'2':1,'3':1,'4':1}
        
    def reset(self):
        self.job_endTime={'j1':0, 'j2':0, 'j3':0, 'j4':0} # job의 끝나는 지점을 등록
        self.machine_endTime={'M1':0,'M2':0,'M3':0,'M4':0} # machine의 끝나는 지점을 등록
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0", 'M4':"j0"}
        self.job_preOperation={'1':1,'2':1,'3':1,'4':1}
        
    def get_fittness(self,scheduling_seq,routing_seq):
        time_list=[]
        for j in range(20):
            job_number = scheduling_seq[j]
            machine = routing_seq[j]
            
            job = "j"+str(job_number)
            jobOp=job+str(self.job_preOperation[str(job_number)])
            self.job_preOperation[str(job_number)] += 1
            
            setup_list = self.s_table[job]
            setup_time=setup_list.loc[self.machine_prejob[machine]]
            start_time = max(self.machine_endTime[machine] ,self.job_endTime[job])
            p_list = self.p_table[machine]
            p_time = p_list.loc[jobOp]
            end_time = start_time +setup_time+p_time
            time_list.append([start_time,end_time])
            self.machine_endTime[machine]=end_time #기계의 끝나는 시간 설정
            self.job_endTime[job]=end_time #job의 끝나는 시간 설정
            self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        all_values = self.machine_endTime.values()
        c_max=max(all_values)
        critical_path = self.Algorithm2(time_list, c_max)
        return c_max, critical_path
    def anneal_eps(self):
        self.eps -=0.01
        self.eps = max(self.eps, 0.2)
    def init_scheduling(self):
        random.shuffle(self.params['job_seq'])
        scheduling_seq=copy.deepcopy(self.params['job_seq'])
        return scheduling_seq
    def init_routing(self, scheduling_seq):
        routing_seq=[]
        for operation in scheduling_seq:
            operation2 = "j" + str(operation) + str(self.job_preOperation[str(operation)])
            self.job_preOperation[str(operation)] += 1
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[operation2]
                if a != 0:
                    routing_seq.append(i)
                    break
        return routing_seq
    
    def Algorithm2(self, time_list, c_max):
        critical_operation=[]
        for i in range(20):
            if time_list[19-i][1] == c_max:
                critical_operation.append(i)
                c_max = time_list[19-i][0]
        return critical_operation
    
    def Ls_operator_scheduling(self, solution):
        coin = random.randint(0,2)
        scheduling_seq, routing_seq, fittness, critical_path = solution
        s_v = copy.deepcopy(scheduling_seq)
        r_v = copy.deepcopy(routing_seq)
        operation_seq = [i for i in range(20)]
        k = random.choice(operation_seq)
        a = operation_seq.pop(k)
        b = random.choice(operation_seq)
        if a > b:
            a,b = b,a
        s_v2 = scheduling_seq[a:b+1]
        r_v2 = routing_seq[a:b+1]
        s_v3 = copy.deepcopy(s_v2)
        if coin == 0:
            s_v3.reverse()
            r_v3 = []
            for i in range(b-a+1):
                for j in range(b-a+1):
                    if s_v3[i] == s_v2[j]:
                        r_v3.append(r_v2[j])
                        s_v2[j] = 0
                        break
            k=0
            for i in range(a,b+1):
                s_v[i] = s_v3[k]
                r_v[i] = r_v3[k]
                k+=1
            c_max, critical_path2 = self.get_fittness(s_v, r_v)
            if c_max < fittness:
                solution = [scheduling_seq, routing_seq, c_max, critical_path2]
            return solution
        elif coin == 1:
            s_v3[0] , s_v3[-1] = s_v3[-1], s_v3[0]
            r_v3 = []
            for i in range(b-a+1):
                for j in range(b-a+1):
                    if s_v3[i] == s_v2[j]:
                        r_v3.append(r_v2[j])
                        s_v2[j] = 0
                        break
            k=0
            for i in range(a,b+1):
                s_v[i] = s_v3[k]
                r_v[i] = r_v3[k]
                k+=1
            c_max, critical_path2 = self.get_fittness(s_v, r_v)
            if c_max < fittness:
                solution = [scheduling_seq, routing_seq, c_max, critical_path2]
            return solution
        elif coin == 2:
            k = s_v3.pop(-1)
            s_v3.insert(0,k)
            r_v3 = []
            for i in range(b-a+1):
                for j in range(b-a+1):
                    if s_v3[i] == s_v2[j]:
                        r_v3.append(r_v2[j])
                        s_v2[j] = 0
                        break
            k=0
            for i in range(a,b+1):
                s_v[i] = s_v3[k]
                r_v[i] = r_v3[k]
                k+=1
            c_max, critical_path2 = self.get_fittness(s_v, r_v)
            if c_max < fittness:
                solution = [scheduling_seq, routing_seq, c_max, critical_path2]
            return solution
    def LS_operator_routing(self, solution):
        coin = random.randint(1, 7)
        scheduling_seq , routing_seq, fittness, critical_path = solution
        if coin == 1:
            search = True
            while search:
                operation_number=0
                operation = random.choice(critical_path)
                for i in range(operation+1):
                    if scheduling_seq[i] == scheduling_seq[operation]:
                        operation_number += 1
                    job_op = "j"+str(scheduling_seq[operation])+str(operation_number)
                random.shuffle(self.params['machine_seq'])
                for i in self.params['machine_seq']:
                    a = self.p_table[i].loc[job_op]
                    if i != routing_seq[operation]:
                        if a != 0:
                            routing_seq2 = copy.deepcopy(routing_seq)
                            routing_seq2[operation] = i
                            search = False
        elif coin == 2:
            operation_number=0
            operation = random.choice(critical_path)
            for i in range(operation+1):
                if scheduling_seq[i] == scheduling_seq[operation]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation]:
                    if a != 0:
                        routing_seq2 = copy.deepcopy(routing_seq)
                        routing_seq2[operation] = i
        elif coin == 3:
            operation = random.choice(critical_path)
            m_c = routing_seq[operation]
            m_seq=[]
            for i in range(20):
                if m_c == routing_seq[i]:
                    m_seq.append(i)
            random.shuffle(m_seq)
            operation2 = m_seq[0]
            operation_number=0
            for i in range(operation2+1):
                if scheduling_seq[i] == scheduling_seq[operation2]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation2])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation2]:
                    if a != 0:
                        routing_seq2 = copy.deepcopy(routing_seq)
                        routing_seq2[operation2] = i
        elif coin == 4:
            operation = random.choice(critical_path)
            m_c = routing_seq[operation]
            m_seq=[]
            for i in range(20):
                if m_c == routing_seq[i]:
                    m_seq.append(i)
            random.shuffle(m_seq)
            operation2 = m_seq[0]
            operation_number=0
            for i in range(operation2+1):
                if scheduling_seq[i] == scheduling_seq[operation2]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation2])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation2]:
                    if a != 0:
                        routing_seq2 = copy.deepcopy(routing_seq)
                        routing_seq2[operation2] = i
        elif coin == 5:
            last_machine = routing_seq[critical_path[-1]]
            m_seq=[]
            for i in range(20):
                if last_machine == routing_seq[i]:
                    m_seq.append(i)
            random.shuffle(m_seq)
            operation2 = m_seq[0]
            operation_number=0
            for i in range(operation2+1):
                if scheduling_seq[i] == scheduling_seq[operation2]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation2])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation2]:
                    if a != 0:
                        routing_seq2 = copy.deepcopy(routing_seq)
                        routing_seq2[operation2] = i
            
        elif coin == 6:
            operation_number=0
            operation = random.randint(0,19)
            for i in range(operation+1):
                if scheduling_seq[i] == scheduling_seq[operation]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation]:
                    if a != 0:
                        routing_seq2 = copy.deepcopy(routing_seq)
                        routing_seq2[operation] = i
        elif coin == 7:
            operation_number=0
            max_time = 10000
            machine = ""
            operation = random.randint(0,19)
            for i in range(operation+1):
                if scheduling_seq[i] == scheduling_seq[operation]:
                    operation_number += 1
            job_op = "j"+str(scheduling_seq[operation])+str(operation_number)
            random.shuffle(self.params['machine_seq'])
            for i in self.params['machine_seq']:
                a = self.p_table[i].loc[job_op]
                if i != routing_seq[operation]:
                    if a < max_time and a != 0:
                        machine = i
                        max_time = a
            routing_seq2 = copy.deepcopy(routing_seq)
            routing_seq2[operation] = machine
        
        c_max, critical_path2 = self.get_fittness(scheduling_seq, routing_seq2)
        coin2 = random.random()
        if c_max < fittness:
            solution[1] = routing_seq2
            solution[3] = critical_path2
            solution[2] = c_max
        elif c_max > fittness and coin2 < self.eps :
            solution[1] = routing_seq2
            solution[3] = critical_path2
            solution[2] = c_max
        return solution
    def sort_population(self, population):
        population.sort(key=lambda x:x[2],reverse=False)
        # todo: fitness를 기준으로 population을 내림차순 정렬하고 반환
        return population
    
    def search(self):
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        all_list = []                    
        for i in range(100):
            scheduling_seq = self.init_scheduling()
            routing_seq = self.init_routing(scheduling_seq)
            self.reset()
            fittness,critical_path = self.get_fittness(scheduling_seq,routing_seq)
            self.reset()
            population.append([scheduling_seq,routing_seq,fittness,critical_path])
            population = self.sort_population(population)
        for i in range(100):
            population = self.sort_population(population)
            coin2 = random.random()
            print(population[0][2])
            for j in range(100):
                if j == 0:
                    s_b = copy.deepcopy(population[0])
                coin = random.randint(0, 1)
                if coin2 < self.eps:
                    s_c = copy.deepcopy(population[j])
                else:
                    s_c = copy.deepcopy(s_b)
                if s_c[2] < population[j][2]:
                    population[j] = copy.deepcopy(s_c)
                else:
                    solution = self.LS_operator_routing(population[j])
                    population[j] = solution
                    self.reset()
                    if solution[2] < s_b[2]:
                        s_b = copy.deepcopy(solution)
            population = self.sort_population(population)
            s_b = copy.deepcopy(population[0])
            for j in range(100):
                solution = self.Ls_operator_scheduling(s_b)
                if solution[2] < s_b[2]:
                    s_b = copy.deepcopy(solution)
                    population[0] = copy.deepcopy(s_b)
                self.reset()
            self.anneal_eps()
            all_list.append(s_b[2])
        return all_list
if __name__ == "__main__":
    jaya = JAYA_FJSP(params)
    sb = jaya.search()
from pylab import plot
plot(sb)
"""
for i in range(100):
    population = sort_population(population)
    if i == 0:
        s_b = copy.deepcopy(population[0])
    if population[0][2] < s_b[2]:
        s_b=copy.deepcopy(population[0])
    LS_operator_routing(population,s_b)
    population = sort_population(population)
    s_b = copy.deepcopy(population[0])
    for x in range(100):
        s_b = Ls_operator_scheduling(s_b)
    print(i,"세대 최고는", s_b[2])
    all_list.append(population[0][2])
population = sort_population(population)
print(population[0])
print(all_list)
"""