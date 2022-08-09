import enum
import this

from pyrsistent import v

#from prometheus_client import container_ip_grouping_key
from sxyAlgo.container import Container
import numpy as np

class Node(object):
    def __init__(self, node_config):
        self.id = node_config["id"]
        self.cpu_capacity = node_config["cpu_capacity"]
        self.mem_capacity = node_config["mem_capacity"]
        self.isMigration = False
        self.cluster = None
        self.containers_bak = None
        self.containers = {}
        self.len = 0
        
        self.cpuPluPredict =None #当前时刻各个vm cpu+predict
        self.memPluPredict = None  #
        
        self.nowPluPredictCost_w = {}# 各个时刻所记录的cost值
        
        self.cpu_sum_w = None #每个时刻cpu之和
        self.mem_sum_w = None
        
        self.CsPluMs = None #每一时刻的pm总开销 pmid: cpu_cost2to2 + b*mem_cost2to2
        self.CsPluMs_migraton =None
        # self.scheduleOut_list={}
        # self.scheduleIn_list={}
        
    def attach(self, cluster):
        self.cluster = cluster

    # TODO 保留
    def add_container(self, container_config):
        #assert container_config.cpu <= self.cpu and container_config.memory <= self.memory 
        container = Container(container_config)
        container_config.node_id = self.id
        container.attach(self)
        self.containers[container.id] = container
    
    def pop(self, container_id):
        container = self.containers.pop(container_id)
        container.node = None
        return container

    def push(self, container):
        self.containers[container.id] = container
        container.attach(self)
    
    def getEveryTimeCpuList(self,clock,w):
        
        nextclock = clock+1
        containers = self.containers
        self.len = len(containers)
        #可能有错
        self.cpuPluPredict = { v.id:np.array(v.cpulist[clock:nextclock]+v.predict(clock,w)[clock]["cpu"]) for v in containers.values() }
        self.memPluPredict = { v.id:np.array(v.memlist[clock:nextclock]+v.predicts[clock]["mem"]) for v in containers.values() for v in containers.values() }
    def cupSumAndMemSum(self):
        
        self.cpu_sum_w = np.sum(np.array([ v for k, v in self.cpuPluPredict.items() ]),axis=0)
        self.mem_sum_w = np.sum(np.array([ v for k,v in self.memPluPredict.items() ]),axis=0)
        
        # print(self.cpu_sum_w)
        # assert 1==0
    def cost_first(self,clock,w,b):
        self.getEveryTimeCpuList(clock,w)
        self.cupSumAndMemSum()
        cpu = self.cpuPluPredict
        mem = self.memPluPredict
        
        csplums = []
        pm_cost = 0
       
        cpusum_w = np.array([ v for v in self.cpu_sum_w])
        memsum_w = np.array([ v for v in  self.mem_sum_w  ])
        
        assert cpusum_w is not None 
        #TODO 这里计算有问题吗？ 怎么总是有负值
        cpukeys = list(cpu.keys())
        lenx = len(cpukeys)-1
        e = -0.000001
        # for j in range(lenx): 
        #     i = cpukeys[j]
            
        #     # print("debug cpu and sum",cpu[i],cpusum_w)
        #     # print("debug mem and sum",mem[i],memsum_w)
        #     cpusum_w -=cpu[i]
        #     memsum_w -= mem[i]
           
        #     for x in range(len(cpusum_w)):
        #         if cpusum_w[x]<0 and cpusum_w[x]>e:
        #             cpusum_w[x]=0
        #         if memsum_w[x]<0 and memsum_w[x]>e:
        #             memsum_w[x]=0
        #     c = ((cpu[i] * cpusum_w ) ) #w个窗口的值
        #     m = ((mem[i] *  memsum_w ))
            
        #     if (c<0).any() or (m < 0).any():
        #         print("debug cpu",cpu,"mem:",mem,"c=",c)
        #         print("debug cpu_sum",self.cpu_sum_w,"mem_sum",self.mem_sum_w,"m=",m)
        #         assert 1==0
        #     csplums.append(c + b * m)
        #     pm_cost += np.sum(csplums[-1] )
        
        
        # OLD #####################################
        cpu_vm = np.array([v for k, v in self.cpuPluPredict.items() ]) # 对应VM在W窗口的所有cpu
        mem_vm = np.array([v for k, v in self.memPluPredict.items() ])
        # print("cpu_vm:",cpu_vm)
        # print("vm",[v for v in self.containers.keys()])
        #cost_t = 0
        csplums = [0.0 for i in range(w)]
        pm_cost = 0
        for i in range(len(cpu_vm)-1): # 每台VM与其他VM两两相乘
            # c = np.sum(cpu_vm[i] * cpu_vm[i+1:])
            # m = np.sum(mem_vm[i] * mem_vm[i+1:])
            for t in range(w):
                c = cpu_vm[i][t] *np.sum( cpu_vm[i+1:,t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:,t])
                #print("c",c)
                csplums[t] += c + b * m
        for t in range(w):
             if csplums[t] > 0.5:
                csplums[t]-=0.5
        pm_cost =np.sum(csplums)
        
        #TODO 这里计算该机器各个时刻的cost总值对吗
        self.CsPluMs = csplums
        # print(self.id,":",self.CsPluMs,np.sum(csplums))
        # if self.id ==2:
        #     assert 1==0
        ###########################################OLD
        # try:
        #     print("debug node>>>",self.CsPluMs[0])
        # except:
        #     print("lenvm:",lenx)
        #     assert 1==0
        self.nowPluPredictCost_w[clock] = pm_cost
        #return  pm_cost
    
    def getnowPluPredictCost(self,clock,w,b):
        if clock not in self.nowPluPredictCost_w or self.nowPluPredictCost_w[clock] is None:
            self.cost_first(clock,w,b)
        return  self.nowPluPredictCost_w[clock] 
    
    def canAddorNot(self,cpu,mem,t):
        return self.cpu_sum[t]+cpu < self.cpu_capacity \
                and  self.mem_sum[t] + mem < self.mem_capacity
     
                
    def migrateOut(self,vmid,t):
        # if t  in self.scheduleOut_list:
        #     self.scheduleOut_list[t].append(vmid)
        # else:
        #     self.scheduleOut_list[t]=[vmid]
        self.pop(vmid)
        return self.memPluPredict[vmid][t]*2
    
    def migrateIn(self,vmid,t):
        # if t  in self.scheduleOut_list:
        #     self.scheduleIn_list[t].append(vmid)
        # else:
        #     self.scheduleIn_list[t]=[vmid]
        self.push(vmid)
    #处理贪心策略的迁移工作
    def afterMigration_cost(self,clock,t,w,b):
       
        cost_t =  {} 
        nextclock = clock+1
        containers = self.containers
        cpu = self.cpuPluPredict = { v.id:np.array(v.cpulist[clock:nextclock]+v.predict(clock,w)[clock]["cpu"]) for v in containers.values() }
        mem = self.memPluPredict= { v.id:np.array(v.memlist[clock:nextclock]+v.predicts[clock]["mem"]) for v in containers.values() for v in containers.values() }
        #换个计算
        # cpu_othersum = { k:[ v1 for k1,v1 in cpu.items() if k1!=k] for k,v in cpu.items() }
        # mem_othersum = { k:[ v1 for k1,v1 in cpu.items() if k1!=k] for k,v in cpu.items() }
        cpu_sum_w = self.cpu_sum_w =  np.sum(np.array([ v for v in cpu.values() ]),axis=0)
        mem_sum_w = self.mem_sum_w =  np.sum(np.array([ v for v in mem.values() ]),axis=0)
        # cpu_t = np.array([ v for v in cpu_sum_w])
        # mem_t =np.array([ v for v in mem_sum_w])
        
        
        # cpu_mem_vector = []
        
        # #TODO 这里计算出负值
        # cpukeys = list(cpu.keys())
        # lenx = len(cpukeys)-1
        # e = -0.0000001
        # for j in range(lenx): 
        #     i = cpukeys[j] 
           
           
        #     cpu_t -= cpu[i]
        #     mem_t -= mem[i]
        #     for x in range(len(cpu_t)):
        #         if cpu_t[x]<0 and cpu_t[x]>e:
        #             cpu_t[x]=0
        #         if mem_t[x]<0 and mem_t[x]>e:
        #             mem_t[x]=0
        #     c = ((cpu[i] * cpu_t ) ) #w个窗口的值
        #     m = ((mem[i] *  mem_t ))
            
        #     if (c<0).any() or (m < 0).any():
        #         print("debug cpu",cpu[i],"mem:",mem[i],"c=",c)
        #         print("debug cpu_sum",cpu_t,"mem_sum",mem_t,"m=",m)
        #         assert 1==0
        #     cpu_mem_vector.append(c + b * m)
        # pm_cost_w = np.sum(cpu_mem_vector,axis=0)

        #OLD ############################################
        cpu_vm = np.array([v for k, v in self.cpuPluPredict.items() ]) # 对应VM在W窗口的所有cpu
        mem_vm = np.array([v for k, v in self.memPluPredict.items() ])
        # print("cpu_vm:",cpu_vm)
        # print("vm",[v for v in self.containers.keys()])
        #cost_t = 0
        csplums = [0.0 for i in range(w)]
        #pm_cost = 0
        for i in range(len(cpu_vm)-1): # 每台VM与其他VM两两相乘
            # c = np.sum(cpu_vm[i] * cpu_vm[i+1:])
            # m = np.sum(mem_vm[i] * mem_vm[i+1:])
            for t in range(w):
                c = cpu_vm[i][t] *np.sum( cpu_vm[i+1:,t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:,t])
                #print("c",c)
                csplums[t] += c + b * m
        for t in range(w):
            if csplums[t] > 0.5:
                csplums[t]-=0.5
            
        #pm_cost =np.sum(csplums)
        cost_t[t] =csplums[t]
        #TODO 这里计算该机器各个时刻的cost总值对吗
        self.CsPluMs_migraton = csplums
        #################################################
        
        #print("\t\t\tpm_cost_w=",len(pm_cost_w),pm_cost_w)
        # cost_t[t] = pm_cost_w[t]
        # #TODO 这里重新计算pm cost值对吗
        # self.CsPluMs_migraton = pm_cost_w
        return cost_t[t]
    
    def backZero(self):
        #让cluster处理
        
        pass
    
    @property
    def cpu(self):
        occupied = 0
        for container in self.containers.values():
            occupied += container.cpu
        return self.cpu_capacity - occupied
   
    @property
    def mem(self):
        occupied = 0
        for container in self.containers.values():
            occupied += container.mem
        
        return self.mem_capacity - occupied
