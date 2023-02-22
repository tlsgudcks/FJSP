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
from collections import Counter
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
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
class JAYA_FJSP():
    def __init__(self, parameters):
        self.eps = 0.5
        self.eps2 = 0.1
        self.eps3 = 0.2
        self.params = {}
        for key, value in parameters.items():
            self.params[key] = value
        self.p_table = pd.read_csv('C:/Users/parkh/git_tlsgudcks/FJSP/data/FJSP_SIM4.csv',index_col=(0)) #job과 operation을 기록한 테이블
        self.s_table = pd.read_csv('C:/Users/parkh/git_tlsgudcks/FJSP/data/FJSP_SETUP_SIM.csv', index_col=(0)) #setup time 테이블
        self.job_endTime={'j1':0, 'j2':0, 'j3':0, 'j4':0, 'j5':0, 'j6':0,'j7':0, 'j8':0, 'j9':0, 'j10':0, 'j11':0, 'j12':0} # job의 끝나는 지점을 등록
        self.machine_endTime={'M1':0,'M2':0,'M3':0,'M4':0} # machine의 끝나는 지점을 등록
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0", 'M4':"j0",'M5':"j0", 'M6':"j0",'M7':"j0", 'M8':"j0"}
        self.job_preOperation={'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'10':1, '11':1, '12':1}
        self.job_max_op = {'1':4, '2':4, '3':4, '4':4, '5':4, '6':4, '7':4, '8':4, '9':4, '10':4, '11':4, '12':4}
        self.s = [0 for x in range(48)]
        self.k = 0
        self.assignment = []
    def reset(self):
        self.job_endTime={'j1':0, 'j2':0, 'j3':0, 'j4':0, 'j5':0, 'j6':0,'j7':0, 'j8':0, 'j9':0, 'j10':0, 'j11':0, 'j12':0} # job의 끝나는 지점을 등록
        self.machine_endTime={'M1':0,'M2':0,'M3':0,'M4':0} # machine의 끝나는 지점을 등록
        self.machine_prejob={'M1':"j0", 'M2':"j0",'M3':"j0", 'M4':"j0",'M5':"j0", 'M6':"j0",'M7':"j0", 'M8':"j0"}
        self.job_preOperation={'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,'10':1, '11':1, '12':1}
        self.s = [0 for x in range(48)]
        self.k = 0
        self.assignment = []
    def operation_check(self, job_type):
        if job_type < 10:
            job = "j0"+str(job_type)
        else:
            job = "j" +str(job_type)
        if self.job_preOperation[str(job_type)] < 10:
            operation = "0"+str(self.job_preOperation[str(job_type)])
        else:
            operation = str(self.job_preOperation[str(job_type)])
        jop = job+operation
        
        return jop
    def gannt_chart(self, population):
        assignment = []
        for j in range(48):
            assignment.append([population[0][j],population[1][j]])
        print(assignment)
        
        plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource']) #간트차트로 보여주기 위한 데이터프레임
        i=0 #간트차트의 인덱싱을 위한 숫자
        j=0
        for job_num,machine in assignment:  #['j11','M2']의 형태에서 잡과 머신을 가져옴
            job = 'j'+str(job_num)        #'j11'의 형태를 j1로 
            machine = "M"+str(machine)
            job_op = self.operation_check(job_num)
            self.job_preOperation[str(job_num)] += 1
            df2_sorted = self.s_table[job] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
            setup_time=df2_sorted.loc[self.machine_prejob[machine]] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
            time = max(self.machine_endTime[machine] ,self.job_endTime[job]) #machine과 job의 순서 제약조건을 지키기 위해 더 큰 값을 설정함
            df_sorted = self.p_table[machine] #p_time테이블에서 현재 machine에 해당하는 열을 가져옴
            p_time = df_sorted.loc[job_op] #해당하는 job과 operation의 시간을 가져옴
            start = datetime.fromtimestamp(time*3600) #포매팅 해줌
            time = time+setup_time # 프로세스타임과 셋업타임을 더해줌
            p_start=datetime.fromtimestamp(time*3600)
            time = time+p_time
            end = datetime.fromtimestamp(time*3600) #끝나는 시간 포매팅
            plotlydf.loc[j] = dict(Task=job, Start=p_start, Finish=end, Resource=machine) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
            if setup_time !=0:
                j+=1
                plotlydf.loc[j] = dict(Task="setup", Start=start, Finish=p_start, Resource=machine) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
            j += 1
            i += 1 #데이터 프레임 인덱싱 증가
            self.machine_endTime[machine]=time #기계의 끝나는 시간 설정
            self.job_endTime[job]=time #job의 끝나는 시간 설정
            self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록      
        self.reset()
        plotlydf2 = plotlydf.sort_values(by=['Resource','Task'], ascending=False)
        fig = px.timeline(plotlydf2, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        fig.show()
    def sco(self, dad, mom):
        dad_ch = copy.deepcopy(dad)
        mom_ch = copy.deepcopy(mom)
        point = random.randint(0,47)
        os_offspring = [-1 for i in range(48)]
        for i in range(point):
            os_offspring[i] = dad_ch[0][i]
            mom_ch[0].remove(os_offspring[i])
        for i in range(48):
            if os_offspring[i] == -1:
                os_offspring[i] = mom_ch[0].pop(0)
        ma_offspring = self.init_MA_LS(os_offspring)
        c_max, critical_machine, flow_time, util = self.get_fittness(os_offspring, ma_offspring)
        offspring = [os_offspring, ma_offspring, c_max, critical_machine, flow_time, util ]
        return offspring
    def jco(self, dad, mom):
        dad_ch = copy.deepcopy(dad)
        mom_ch = copy.deepcopy(mom)
        job_list = [x for x in range(1,13)]
        for i in range(1,13):
            coin = random.random()
            if coin >0.5:
                job_list.remove(i)
        os_offspring = [-1 for i in range(48)]
        for i in range(48):
            if dad_ch[0][i] in job_list:
                os_offspring[i] = dad_ch[0][i]
                mom_ch[0].remove(dad_ch[0][i])
        for i in range(48):
            if os_offspring[i] == -1:
                os_offspring[i] = mom_ch[0].pop(0)
        ma_offspring = self.init_MA_LS(os_offspring)
        c_max, critical_machine, flow_time, util = self.get_fittness(os_offspring, ma_offspring)
        offspring = [os_offspring, ma_offspring, c_max, critical_machine, flow_time, util ]
        return offspring
    def aco(self, dad, mom):
        dad_ch = copy.deepcopy(dad)
        mom_ch = copy.deepcopy(mom)
        os_offspring = copy.deepcopy(dad_ch[0])
        ma_offspring = [-1 for i in range(48)]
        for i in range(48):
            for j in range(48):
               if os_offspring[i] == mom_ch[0][j]:
                   mom_ch[0][j] = 0
                   ma_offspring[i] = mom_ch[1][j]
                   break
        c_max, critical_machine, flow_time, util = self.get_fittness(os_offspring, ma_offspring)
        offspring = [os_offspring, ma_offspring, c_max, critical_machine, flow_time, util ]
        return offspring
    def get_fittness(self,scheduling_seq,routing_seq):
        time_list=[]
        p_time_list = []
        for j in range(48):
            job_number = scheduling_seq[j]
            machine = routing_seq[j]
            job = "j"+str(job_number)
            jobOp = self.operation_check(job_number)
            self.job_preOperation[str(job_number)] += 1
            
            setup_list = self.s_table[job]
            setup_time=setup_list.loc[self.machine_prejob[machine]]
            start_time = max(self.machine_endTime[machine] ,self.job_endTime[job])
            p_list = self.p_table[machine]
            p_time = p_list.loc[jobOp]
            p_time_list.append(p_time)
            end_time = start_time +setup_time+p_time
            time_list.append([start_time,end_time])
            self.machine_endTime[machine]=end_time #기계의 끝나는 시간 설정
            self.job_endTime[job]=end_time #job의 끝나는 시간 설정
            self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        all_values = self.machine_endTime.values()
        all_values2 = self.job_endTime.values()
        all_time = sum(all_values)
        flow_time = sum(all_values2)
        value_add_time = sum(p_time_list)
        util = value_add_time / all_time
        c_max=max(all_values)
        all_values = list(all_values)
        k=0
        critical_machine=""
        for i in range(4):
            if all_values[i] > k:
                k=all_values[i]
                critical_machine = str(i+1)
        return c_max, critical_machine, flow_time, util
    def get_fittness2(self,scheduling_seq,routing_seq):
        time_list=[]
        p_time_list = []
        for j in range(48):
            job_number = scheduling_seq[j]
            machine = routing_seq[j]
            machine = "M"+str(machine)
            job = "j"+str(job_number)
            jobOp = self.operation_check(job_number)
            self.job_preOperation[str(job_number)] += 1
            
            setup_list = self.s_table[job]
            setup_time=setup_list.loc[self.machine_prejob[machine]]
            start_time = max(self.machine_endTime[machine] ,self.job_endTime[job])
            p_list = self.p_table[machine]
            p_time = p_list.loc[jobOp]
            p_time_list.append(p_time)
            end_time = start_time +setup_time+p_time
            time_list.append([start_time,end_time])
            self.machine_endTime[machine]=end_time #기계의 끝나는 시간 설정
            self.job_endTime[job]=end_time #job의 끝나는 시간 설정
            self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        all_values = self.machine_endTime.values()
        all_values2 = self.job_endTime.values()
        all_time = sum(all_values)
        flow_time = sum(all_values2)
        value_add_time = sum(p_time_list)
        util = value_add_time / all_time
        c_max=max(all_values)
        all_values = list(all_values)
        k=0
        critical_machine=""
        for i in range(4):
            if all_values[i] > k:
                k=all_values[i]
                critical_machine = str(i+1)
        return c_max, critical_machine, flow_time, util
    def anneal_eps(self):
        self.eps -=0.001
        self.eps2 +=0.001
        self.eps2 = min(self.eps2, 0.6)
        self.eps3 +=0.002
        self.eps3 = min(self.eps3, 0.8) 
    def init_OS(self):
        random.shuffle(self.params['job_seq'])
        scheduling_seq=copy.deepcopy(self.params['job_seq'])
        return scheduling_seq
    def init_MA_Random(self, scheduling_seq): # 이게 걍 랜덤하게 초기화
        routing_seq=[]
        for operation in scheduling_seq:
            operation2 = self.operation_check(operation)
            self.job_preOperation[str(operation)] += 1
            for i in self.machine_endTime:
                a = self.p_table[i].loc[operation2]
                if a != 0:
                    routing_seq.append(i)
                    break
        return routing_seq
    def init_MA_LS(self, scheduling_seq): #이건 SPT초기화
        routing_seq=[]
        for operation in scheduling_seq:
            operation2 = self.operation_check(operation)
            self.job_preOperation[str(operation)] += 1
            machine = self.least_time_machine(operation,operation2)
            routing_seq.append(machine)
        self.reset()
        return routing_seq
    def AAM(self, offspring): #local하게 좋은 
        coin = random.randint(1, 1)
        scheduling_seq ,routing_seq, fittness, critical_machine, flow_time, util = offspring
        if coin == 1:
            routing_seq2 = copy.deepcopy(routing_seq)
            critical_machine_index = []
            machine = "M"+str(critical_machine)#과부하 걸린 기계
            for i in range(48): #과부하 걸린 기계의 순서를 저장해놓음
                if machine == routing_seq[i]:
                    critical_machine_index.append(i) #[1,13,20,21,24,30,38]
            random.shuffle(critical_machine_index)   
            job = scheduling_seq[critical_machine_index[0]] # 24의 7이었음
            k=0
            for i in range(critical_machine_index[0]+1): #24까지 돌면서 7이 몇번나오나
                if job == scheduling_seq[i]:
                    k+=1
            if job>9:
                job_op = "j"+str(job)+"0"+str(k) #72
            else:
                job_op = "j0"+str(job)+"0"+str(k) #72
            machine = self.random_machine(job, job_op)
            routing_seq2[critical_machine_index[0]] = machine
            self.reset()
            fittness, critical_machine, flow_time, util = self.get_fittness(scheduling_seq, routing_seq2)
            solution = [scheduling_seq, routing_seq2, fittness, critical_machine,flow_time,util]
        return solution
    def least_time_machine(self, job,operation2): #그 시점에서 가장 작은 machine 선택해주는거
        best_machine = ""
        max_endTime=10000
        for i in self.machine_endTime:
            time = max(self.machine_endTime[i] ,self.job_endTime["j"+str(job)])
            df_sorted = self.p_table[i]
            p_time = df_sorted.loc[operation2]
            if p_time != 0:
                df2_sorted = self.s_table["j"+str(job)]
                setup_time=df2_sorted.loc[self.machine_prejob[i]]
                start = time
                end = start+setup_time+p_time
                if max_endTime>end:
                    max_endTime = end
                    best_machine = i
        self.machine_endTime[best_machine]=max_endTime #기계의 끝나는 시간 설정
        self.job_endTime[job]=max_endTime #job의 끝나는 시간 설정
        self.machine_prejob[best_machine] = "j"+str(job) #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        return best_machine
    def random_machine(self, job,operation2): #그 시점에서 가장 작은 machine 선택해주는거
        random_machine_list = []
        for i in self.machine_endTime:
            df_sorted = self.p_table[i]
            p_time = df_sorted.loc[operation2]
            if p_time != 0:
                random_machine_list.append(i)
        random.shuffle(random_machine_list)
        return random_machine_list[0]
    def Two_Point_OS_LS(self, solution): #LS 세가지
        coin = random.randint(0,2)
        scheduling_seq, routing_seq, fittness, critical_machine, flow_time, util = solution
        s_v = copy.deepcopy(scheduling_seq)
        r_v = copy.deepcopy(routing_seq)
        operation_seq = [i for i in range(48)]
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
            c_max, critical_machine2,flow_time, util = self.get_fittness(s_v, r_v)
            solution = [s_v ,r_v, c_max, critical_machine2, flow_time, util]
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
            c_max, critical_machine2,flow_time, util = self.get_fittness(s_v, r_v)
            solution = [s_v ,r_v, c_max, critical_machine2, flow_time, util]
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
            c_max, critical_machine2,flow_time, util = self.get_fittness(s_v, r_v)
            solution = [s_v ,r_v, c_max, critical_machine2, flow_time, util]
            return solution
    def step(self, a):
        machine = a+1
        self.s[self.k] = machine
        reward = self.return_reward()
        done = self.is_done()
        self.k+=1
        return self.s, reward, done
    def is_done(self):
        done = False
        if self.s[len(self.s)-1] == 0:
            done = False
        else:
            done = True
        return done
    def return_reward(self):
        job_num = self.assignment[self.k]
        machine = "M"+str(self.s[self.k])         #'j11'의 형태를 j1로
        job = 'j'+str(job_num)        #'j11'의 형태를 j1로 
        job_op = self.operation_check(job_num)
        self.job_preOperation[str(job_num)] += 1
        df2_sorted = self.s_table[job] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
        setup_time=df2_sorted.loc[self.machine_prejob[machine]] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        last_work_finish_time = self.machine_endTime[machine]
        time = max(self.machine_endTime[machine] ,self.job_endTime[job]) #machine과 job의 순서 제약조건을 지키기 위해 더 큰 값을 설정함
        df_sorted = self.p_table[machine] #p_time테이블에서 현재 machine에 해당하는 열을 가져옴
        p_time = df_sorted.loc[job_op] #해당하는 job과 operation의 시간을 가져옴
        start = datetime.fromtimestamp(time*3600) #포매팅 해줌
        time = time+setup_time # 프로세스타임과 셋업타임을 더해줌
        p_start=datetime.fromtimestamp(time*3600)
        time = time+p_time
        end = datetime.fromtimestamp(time*3600) #끝나는 시간 포매팅
        self.machine_endTime[machine]=time #기계의 끝나는 시간 설정
        self.job_endTime[job]=time #job의 끝나는 시간 설정
        self.machine_prejob[machine] = job #현재 어떤 machine에서 어떤 job을 수행했는지 기록
        reward = 30-p_time
        return reward
    def random_action(self):
        #인덱스는 self.k
        random_machine_list = []
        job = self.assignment[self.k]
        jop = self.operation_check(job)
        for i in self.machine_endTime:
            df_sorted = self.p_table[i]
            p_time = df_sorted.loc[jop]
            if p_time != 0:
                random_machine_list.append(i)
        random.shuffle(random_machine_list)
        random_machine = random_machine_list[0]
        random_action = int(random_machine[1])-1
        return random_action
    def replacement_operator2(self, population, offsprings):
        result_population = population[:]
        for i in range(5):
            fitness = offsprings[i][2]
            for j in range(100):
                if population[j][2] > fitness:
                    result_population[j] = copy.deepcopy(offsprings[i])
                    break
        return result_population
    def sort_population(self, population):
        population.sort(key=lambda x:x[2],reverse=False)
        # todo: fitness를 기준으로 population을 내림차순 정렬하고 반환
        return population
    def selection_operater(self, population):
        mom_ch = []
        dad_ch = []
        fitness_population=copy.deepcopy(population) #리스트 복사
        total_score = 0 #유전자의 적합도 총합 계산
        sum_fitness = 0 #0부터 적합도들의 합을 더함 이것이 k보다 클 시 그 해를 선택
        #유전자별 적합도를 새로운 리스트에 입력
        for i in range(len(population)):
            fitness_population[i][2]=abs((population[-1][2]-population[i][2])-(population[0][2]-population[-1][2])/(3-1))
        #(Cw-Ci)+(Cb-Cw)/(k-1)
        
        #유전자의 적합도 총합계산
        for i in range(len(fitness_population)):
            total_score=fitness_population[i][2]+total_score
        total_score=int(total_score)
        #랜덤함수 호출        
        k=random.uniform(0, total_score)
        #룰렛 돌리기
        for i in range(len(fitness_population)):
            sum_fitness=sum_fitness+fitness_population[i][2]
            if sum_fitness>=k:
                dad_ch=population[i]
                fitness_population.pop(i)
                break
        #적합도 총합을 다시계산해줌, 더했던 적합도들의 합도 다시 계산
        sum_fitness=0
        total_score=0
        for i in range(len(fitness_population)):
            total_score=fitness_population[i][2]+total_score
        total_score=int(total_score)
        #랜덤함수 실행
        k2=random.uniform(0, total_score)
        #룰렛 돌리기
        for i in range(len(fitness_population)):
            sum_fitness=sum_fitness+fitness_population[i][2]
            if sum_fitness>=k2:
                mom_ch=population[i]
                break
        return mom_ch, dad_ch
    def replacement_operator(self, population, offsprings):
        # todo: 생성된 자식해들(offsprings)을 이용하여 기존 해집단(population)의 해를 대치하여 새로운 해집단을 return
        """
        세대형 유전 알고리즘 사용
        해집단 내에서 가장 품질이 낮은 해를 대치하는 방법 사용(엘리티즘)
        """
        result_population = []
        for i in range(5):
            population.pop()
        for i in range(5):
            population.append(offsprings[i])
        result_population=population[:]
        return result_population
    def print_average_fitness(self, population):
        # todo: population의 평균 fitness를 출력
        population_average_fitness = 0
        total_population=0 
        for i in range(100):
            total_population=total_population+population[i][2]
        population_average_fitness=total_population/100
        print("population 평균 fitness: {}".format(population_average_fitness))
        return population_average_fitness
    def search(self):
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        all_list = []                    
        for i in range(100):
            scheduling_seq = self.init_OS()
            routing_seq_m = self.init_MA_LS(scheduling_seq)
            self.reset()
            fittness,critical_machine,flow_time,util = self.get_fittness(scheduling_seq,routing_seq_m)
            self.reset()
            population.append([scheduling_seq,routing_seq_m,fittness,critical_machine, flow_time, util])
        population = self.sort_population(population)
        result=[]
        while True:
            offsprings = []
            count_end=0 #동일 갯수
            for i in range(5):
                mom_ch, dad_ch = self.selection_operater(population)
                self.reset()
                coin = random.randint(0,1)
                if coin == 0 :
                    offspring = self.sco(dad_ch, mom_ch)
                    self.reset()
                elif coin == 1:
                    offspring = self.jco(dad_ch, mom_ch)
                    self.reset()
                coin2 = random.random()
                if coin2 < 0.15:
                    offspring = self.Two_Point_OS_LS(offspring)
                offsprings.append(offspring)
            population = self.replacement_operator(population, offsprings)
            self.reset()
            population = self.sort_population(population)
            for i in range(5):
                mom_ch, dad_ch = self.selection_operater(population)
                self.reset()
                coin = random.randint(0,2)
                if coin == 0 :
                    offspring = self.sco(dad_ch, mom_ch)
                    self.reset()
                elif coin == 1:
                    offspring = self.jco(dad_ch, mom_ch)
                    self.reset()
                else:
                    offspring = self.aco(dad_ch, mom_ch)
                    self.reset()
                coin2 = random.random()
                if coin2 < 0.15:
                    offspring = self.Two_Point_OS_LS(offspring)
                elif 0.15< coin2 <0.25:
                    offspring = self.AAM(offspring)
            population = self.replacement_operator(population, offsprings)
            self.reset()
            population = self.sort_population(population)
            print('현재 세대',generation, '최고 해', population[0][2])
            avg = self.print_average_fitness(population)
            result.append([population[0][2],avg])
            self.anneal_eps()
            generation=generation+1
            pop_list=[]
            for i in range(100):
                pop_list.append(population[i][2])
            a = Counter(pop_list)
            end_number_list=[]
            for k in a.values():
                end_number_list.append(k)
            end_number = max(end_number_list)
            if end_number >= 90:
                self.reset()
                print(generation)
                break
            if generation == 1000:
                print(generation)
                break
        
        return population, result
