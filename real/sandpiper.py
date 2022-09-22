#!/usr/bin/env python
# coding: utf-8
from time import time
from typing import overload
from utl import ResourceUsage,isAllUnderLoad

import numpy as np

def Sandpiper_algo(x_t0, cpu_t0, mem_t0,CPU_MAX, MEM_MAX):
    """
    sandpiper调度算法触发机制：每固定时间段检测是否有过载机器（即某机器的某资源量>75%），若有则触发
    Args:
        x_t0 (numpy.ndarray): t0时刻VM部署情况 N*M (vm N个, pm M个)
        cpu_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        mem_t0 (numpy.ndarray): N个VM当前时刻的cpu需求量
        CPU_MAX (int): cpu资源使用上限
        MEM_MAX (int): mem资源使用上限

    Returns:
        ndarry: t1时刻的VM部署情况x_t1 (t1 = t0 + 1)
    """
    f = isAllUnderLoad(x_t0, cpu_t0, mem_t0, CPU_MAX, MEM_MAX)
    if f :
        print("no schedule")
        return x_t0
    # 计算当前各PM和VM的Vol值及VSR值
    CPU_t0 = ResourceUsage(cpu_t0, x_t0)
    MEM_t0 = ResourceUsage(mem_t0, x_t0)
    Vol_pm = 10000 / ((100 - CPU_t0) * (100 - MEM_t0)) # 注意每PM/VM的资源需求要<100%
    Vol_vm = 10000 / ((100 - cpu_t0) * (100 - mem_t0)) # 1*N矩阵
    VSR = Vol_vm/ mem_t0 # 1*N矩阵
    # 机器按Vol值按序排序，存储机器号
    pm_asc = Vol_pm.argsort()
    pm_desc = pm_asc[::-1]
    #print('pm_desc',pm_desc)
    placement = x_t0.copy() # 初始化
    # 按序对每台机器做迁出
    pm_asc.astype(int)
    pm_desc.astype(int)
    migs_inc_outToin = {}
    pmout = {}
    pmin={}
    #print(f'\t to schedule pm num is {len(pm_desc)}')
    idx=0
    motivation = {}
    test_cpu = set()
    test_mem = set()
    metric_pmout = set()
    metric_pmin = set()
    for pm_outs in pm_desc:
        pm_out=int(pm_outs)
        #print('\t\t sand piper',idx)
        idx+=1
        if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载
            continue
        # 将每台机器上的VM降序排序，存储VM号
        vm_in_pm = np.where(x_t0[:, pm_out] == 1)[0] # 该机器上VM的VM号
        VSR_in_pm = VSR[vm_in_pm] # 这些VM的VSR
        #print(vm_in_pm,VSR_in_pm)
        vm_VSR = np.array([vm_in_pm, VSR_in_pm]) # 二维数组，第一行为VM号，第二行为VM对应VSR值
        vm_asc = vm_VSR.T[np.lexsort(vm_VSR)].T # 按照VSR升序排序
        #vm_asc = vm_VSR[:,vm_VSR[1].argsort()]
        vm_desc = vm_asc[0, ::-1] # 获取降序排序后的VM号
        vm_desc.astype(int)
        for vms in vm_desc: # 从VSR最大的VM开始被迁移
            vm = int(vms)
            if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载
                
                break
            for pm_inx in pm_asc: # 从Vol最小的开始迁入
                pm_in = int(pm_inx)
                if CPU_t0[pm_in] + cpu_t0[vm] <= CPU_MAX and MEM_t0[pm_in] + mem_t0[vm] <= MEM_MAX: # 有机器放得下
                    metric_pmout.add(pm_out)
                    metric_pmin.add(pm_in)
                    s = time()
                  
                    placement[vm][pm_out] = 0 # 迁出机器
                    CPU_t0[pm_out] = CPU_t0[pm_out] - cpu_t0[vm]
                    MEM_t0[pm_out] = MEM_t0[pm_out] - mem_t0[vm]
                    
                    placement[vm][pm_in] = 1 # 迁入机器
                    CPU_t0[pm_in] = CPU_t0[pm_in] + cpu_t0[vm]
                    MEM_t0[pm_in] = MEM_t0[pm_in] + mem_t0[vm]
                    if pm_out in pmout:
                        pmout[pm_out].append(vm)
                    else:
                        pmout[pm_out]=[vm]
                    if pm_in in pmin:
                        pmin[pm_in].append(vm)
                    else:
                        pmin[pm_in]=[vm]
                    migs_inc_outToin[vm]=(pmout,pmin)
                    
    
                    
                    break
      
        # 循环结束条件是没有机器过载
        if (CPU_t0 <= CPU_MAX).all() and (MEM_t0 <= MEM_MAX).all():
            #f = isAllUnderLoad(placement, cpu_t0, mem_t0, CPU_MAX, MEM_MAX)
            #print(f'\t\tall pm is underload ?={f}')
            break
    # underhead = (CPU_t0 <= CPU_MAX).all() and (MEM_t0 <= MEM_MAX).all()
    # f = isAllUnderLoad(placement, cpu_t0, mem_t0, CPU_MAX, MEM_MAX)
    #print(f'\t\t\tunderload?= {f} underhead?= {underhead}')
    # 循环结束条件是没有机器过载   
    
    
    return placement