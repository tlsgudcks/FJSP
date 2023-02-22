# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 00:09:48 2023

@author: parkh
"""
import copy
import random


job = [1,1,1,1,1]
d_os = [1,2,3,2,3,4,1,2,3,4,4,5,5,2]
d_ma = [2,1,2,3,2,1,2,3,2,3,2,1,2,3]
dad = [d_os,d_ma]
m_os = [2,4,5,4,3,2,1,1,3,5,4,3,2,2]
m_ma = [2,2,3,1,2,3,2,2,1,2,3,2,1,2]
mom = [m_os,m_ma]
def sco(dad, mom):
    dad_ch = copy.deepcopy(dad)
    mom_ch = copy.deepcopy(mom)
    point = random.randint(0,13)
    print(point)
    os_offspring = [-1 for i in range(14)]
    ma_offspring = [-1 for i in range(14)]
    for i in range(point):
        os_offspring[i] = dad_ch[0][i]
        mom_ch[0].remove(os_offspring[i])
    for i in range(14):
        if os_offspring[i] == -1:
            os_offspring[i] = mom_ch[0].pop(0)
    return os_offspring
def jco(dad, mom):
    dad_ch = copy.deepcopy(dad)
    mom_ch = copy.deepcopy(mom)
    job_list = [x for x in range(1,6)]
    for i in range(1,6):
        coin = random.random()
        if coin >0.5:
            job_list.remove(i)
    os_offspring = [-1 for i in range(14)]
    ma_offspring = [-1 for i in range(14)]
    for i in range(14):
        if dad_ch[0][i] in job_list:
            os_offspring[i] = dad_ch[0][i]
            mom_ch[0].remove(dad_ch[0][i])
    for i in range(14):
        if os_offspring[i] == -1:
            os_offspring[i] = mom_ch[0].pop(0)
    return os_offspring
def aco(dad, mom):
    dad_ch = copy.deepcopy(dad)
    mom_ch = copy.deepcopy(mom)
    os_offspring = copy.deepcopy(dad_ch[0])
    ma_offspring = [-1 for i in range(14)]
    for i in range(14):
        for j in range(14):
           if os_offspring[i] == mom_ch[0][j]:
               mom_ch[0][j] = 0
               ma_offspring[i] = mom_ch[1][j]
               break


sco(dad,mom)
jco(dad,mom)
aco(dad,mom)

