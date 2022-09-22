

import numpy as np
from sxyAlgo.cluster import Cluster 
#from arima.predict import arimas
from time import time 
from pyDOE import lhs
#from statsmodels.tsa.arima.model import ARIMAResults
from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass
class Algorithm_sxy(Algorithm):
    def __call__(self, cluster:Cluster,now):
        
        self.cluster = cluster
        self.params = {"w":3,"z":range(20),"k":5,"u":0.8,"v":0.4,"a":0.004,
                       "b":0.0025,"y":0.25,"N":len(cluster.containers),
                       "M":len(cluster.nodes)}
        
        value,eval_bal,eval_mig = self.schedule(now)
        return value,eval_bal,eval_mig
   
    def schedule(self,now):
        start = time()
        params = self.params
        #怎么保证每次循环的随机都不一样呢
        min_z ,eval_bal,eval_mig,value = self.SchedulePolicy( params["z"], params["k"], params["w"], params["v"], 
                                                             params["M"],params["a"],params["b"],params["y"],now)
        
        # sxytime = time()
        # sxybal = self.CostOfnodeLoadBalanceSimplify(0,params["b"],params["N"],params["M"])
        after =  time() 
        # print("计算时间",after-sxytime,"sxy bal :",sxybal,"my bal",eval_bal)
        
        if min_z!=-1:
            print("at ",now,"花费了",after-start,"s metric=",min_z ,eval_bal,eval_mig,value)
        else:
             print("at ",now,"没有最优，总共花费了",after-start ,eval_bal,eval_mig,value)
        return value,eval_bal,eval_mig

    def SchedulePolicy(self,Z, K, W, v, M, a, b,y,now):
        cluster = self.cluster
        lenx = len(cluster.containers[0].cpulist)
        s = time()
        # TODO debug
        # print(cluster.nodes.keys())
        # assert 1==0
        cost_min,balfirst = cluster.cost_all_pm_first(now,W,b)
        print("算法开始执行 计算 cost_min    消耗了 %.2f s, cost_min = %.3f " % (time()-s,cost_min))
        s = time()
        over,under = self.findOverAndUnder(cluster,b,y,0)
        
        print("算法开始执行 计算over & under 消耗了 %.2f s, length of over = %d  length of under = %d " % (time()-s,len(over),len(under))) 
        
        CPU_t,MEM_t=None,None
        
        candidate_copy = {}
        min_z ,balf,migf,valuef= -1,balfirst,0,balfirst
        for z in Z:
            cost = 0 # 当前W内全套配置方案下的总成本
            findOV = (over,under)
            bal,mig,value= 0,0,0
            CPU_t,MEM_t=None,None
            #print("\tz=",z)
            for t in range(W):
                if t==1 and now+W-1 >= lenx:
                    break
                candidate = {}
                if t!=0 and flag :
                    findOV= self.findOverAndUnder(cluster,b,y,t)
                    #self.writeCompare("over_t=1:",findOV[0],"runthis_over.txt")
                flag = False
                
                #print("over:",len(findOV[0]),"under",len(findOV[1]))
                for k in range(K):
                    #TODO
                    CPU_t,MEM_t,flag = self.RandomGreedySimplify_new(M, a, b, v,t,findOV,candidate,CPU_t,MEM_t)
                    if flag:
                        #print("\t\tt=",t,"k=",k,"\tcandidate len=",len(candidate))
                        break
                    candidate.clear()
                if t==0:
                    candidate_copy[z] = candidate
                if len(candidate)==0 or flag==False:
                    #TODO 
                    cost+=cluster.NoMigration(t)
                    if t==0:
                        mig ,bal,value= 0,cost,cost
                    continue
                migx,balx,valuex= cluster.costForMigration(candidate,now,t,W,b,a,M)
                cost+=valuex
                if t==0:
                    mig ,bal,value =migx,balx,valuex
                if cost > cost_min:
                    break
            print(f"find cost greedily z={z} cost={cost}")
            if cost < cost_min: # 选择总开销低于完全不迁移的开销且最小的
                cost_min = cost
                min_z ,balf,migf,valuef= z, bal,mig,value
            cluster.backZero(z,now,W)
        
        migNum = cluster.freshStructPmVm(candidate_copy,min_z,now)
        #balxxx = cluster.
        return  min_z ,balf,migf,valuef
    
    # def writeCompare(self,name,data,files):
    #     filename = "/hdd/lsh/Scheduler/runthis/compare/"+files
    #     file = open(filename,'a')
        
    #     file.write(name)
    #     file.write(str(data))
    #     file.write("\n")
    #     file.close()
    
    def findOverAndUnder(self,cluster:Cluster,b,y,t):
        #这个cpusum 会被迁移影响么 已经在costForMigration更新了
        #print("cluster.cpusum",cluster.cpusum[0])
       
        pm_cpu_t = { k: cpusumlist[t] for k,cpusumlist in cluster.cpusum.items() }# cpusumlist各个时刻的资源综合
        pm_mem_t = { k: cpusumlist[t] for k,cpusumlist in cluster.memsum.items() }
        
        ##test debug
        # CPU_t = { k: cpusumlist[t] for k,cpusumlist in sorted(cluster.cpusum.items(),key=lambda x:x[0]) }
        # CPU_t = np.array(list(CPU_t.values()))
        # self.writeCompare("findoverandunder CPU_t:",CPU_t,"runthis_CPU_t.txt")
        ##test debug
        
        params = self.params
        allcpuValue = np.array(list(pm_cpu_t.values()))
        allmemValue = np.array(list(pm_mem_t.values()))
        avg_CPU = np.sum(allcpuValue) / params["M"] # 计算当前时刻最负载均衡情况下每台机器应承载的资源均量
        avg_MEM = np.sum(allmemValue) / params["M"] 
        max_CPU = max(allcpuValue) # 当前最大值
        max_MEM = max(allmemValue)

        thr_CPU = y * (max_CPU - avg_CPU) + avg_CPU
        thr_MEM = y * (max_MEM - avg_MEM) + avg_MEM
        
        # TODO debug
        #print("debug cluster.vm_cpu in findOver ",type(cluster.vm_cpu),cluster.vm_cpu)
        cpu_t = cluster.vm_cpu[:,t] 
        mem_t = cluster.vm_mem[:,t]
        cpumem = np.vstack((cpu_t, mem_t)).T # 合并为一个二维数组
        cpumem_desc = cpumem[np.lexsort(cpumem[:,::-1].T)] # 按照cpu的大小降序排序
        #TODO
        thresh_out = (thr_CPU ** 2 + b * thr_MEM ** 2) / 2
        #thresh_out = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2
        thresh_in = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2 # 判断迁入机器候选集的标准，初始化
        cpu_sum = 0
        mem_sum = 0
        for i in cpumem_desc: # 对数组中的每一行，即每一个cpu-mem对
            cpu_sum = cpu_sum + i[0]
            mem_sum = mem_sum + i[1]
            if cpu_sum < avg_CPU and mem_sum < avg_MEM: # 还没达到均值
                temp = (i[0] ** 2 + b * i[1] ** 2) / 2
                thresh_in = thresh_in - temp
            else: # cpu或mem之和大于等于均值，则结束循环
                temp = ((avg_CPU - cpu_sum + i[0]) ** 2 + b * (avg_MEM - mem_sum + i[1]) ** 2) / 2
                thresh_in = thresh_in - temp
                break
        
       
        allVmCsPluMs =cluster.pm_cost  #TODO 已经在costForMigration更新了
        #TODO debug
        try:
            bal = { pmid:v[t] for pmid,v in allVmCsPluMs.items()}  # [w1_sum,w2_sum]
        except:
            print("debug wron t = ",t)
            for k,v in allVmCsPluMs.items():
                print(k,v)
            #print("debug :",len(allVmCsPluMs[0]),allVmCsPluMs[0],"\n t = ",t)
            print("debug wrong")
            exit()
        pmids = np.array(list(bal.keys()))
        v = np.array(list(bal.values()))
        # print(" in Algorithm_sxy.py line 149 ------ thresh_out = ",thresh_out,"thresh_in = ",thresh_in)
        # print(f"bal.values = {v}")
        overv = np.where(v > thresh_out)[0] # 迁出候选集
        over = pmids[overv]
        underv = np.where(v < thresh_in)[0] # 迁入候选集
        under = pmids[underv]
        return over,under
        
    def RandomGreedySimplify_new(self,M, a, b, v,t,findOV,candidate,CPU_t=None,MEM_t=None):
        cluster = self.cluster
        nodes = cluster.nodes
        cpusum =cluster.cpusum
        memsum=cluster.memsum
        
        #TODO debug cpu_t
        cpu_t = list(cluster.vm_cpu[:,t]) # 提取当前时刻各VM的cpu需求量
        mem_t = list(cluster.vm_mem[:,t])
        #print("at t=",t,"len of cluster.vm_cpu ",len(cpu_t),len(cluster.containers),len(cluster.vm_cpu))
        #assert len(cpu_t) == len(cluster.containers)

        over,under= findOV
        if CPU_t is None or MEM_t is None:
            CPU_t = { k: cpusumlist[t] for k,cpusumlist in sorted(cpusum.items(),key=lambda x:x[0]) }# cpusumlist各个时刻的资源综合
            MEM_t  = { k: memsumlist[t] for k,memsumlist in sorted(memsum.items(),key=lambda x:x[0]) }
            CPU_t = np.array(list(CPU_t.values()))
            MEM_t = np.array(list(MEM_t.values()))
        
        #max_under = np.max(under)
        for s in over:
            
            nodethis = nodes[s]
            containers = nodethis.containers
            mig_candi_s = np.array([ x  for x in containers.keys() ])# 能被迁走的VM候选集

            # max_mig = np.max(mig_candi_s)
            # min_mig = np.min(mig_candi_s)

            samples=np.ceil(v*len(mig_candi_s))
            samples = int(samples)
            lhd = lhs(1, samples) # 拉丁超立方抽样，输出[0,1]
            mig_loc = lhd * len(mig_candi_s)
            mig_loc = mig_loc[:,0].astype(int) # 即要被迁移的contaienr的id在候选集中的位置

            # TODO debug
            mig = np.unique(mig_candi_s[mig_loc]).astype(int)  # 要被迁移的contaienr的id，去掉重复值
            #print("debug mig in RandomGreedySimplify_new mig_loc = ",mig_loc,"mig_candi_s = ",mig_candi_s)#,"mig=",mig_candi_s[mig_loc])
            
            # TODO debug index out bounder
            #print("debug mig in RandomGreedySimplify_new = ",mig,"cpu_t len=",len(cpu_t))
            
            # 对每个迁移VM贪心选择最优迁入机器
            for m in mig:
                
                destination = s # 目标机器初始化为原本所在的机器
                m = int(m)
                
                # TODO debug mig cpu mem
                # print("debug in bal_d_cpu " ,CPU_t,MEM_t,CPU_t[under])
                # print(f"over = {over}, mig = {mig} under = {under}")
                #print((t,m,s,max_under,len(CPU_t),len(cpu_t),max_mig,min_mig))
                print(t,m,s,len(CPU_t),len(cpu_t))
                bal_d_cpu = cpu_t[m] * (CPU_t[s] - cpu_t[m] - CPU_t[under]) # 该VM资源量*（原机器上除该VM之外的资源总量-目标机器上原本的资源总量）
                bal_d_mem = mem_t[m] * (MEM_t[s] - mem_t[m] - MEM_t[under])

                bal_d = np.array(bal_d_cpu + b * bal_d_mem)
                mig_m = np.array( a * (M-1) * mem_t[m])
                idx = np.array(np.where(bal_d > mig_m)[0])
                lendx = len(idx)
                if lendx==0:
                    continue
                allmetric = bal_d
                #print("debug >>> ",allmetric[0])
                tmps = {under[idx[i]]: allmetric[idx[i]] for i in range(lendx)}
                candiUnder = [k for k,v in sorted(tmps.items(),key=lambda x:x[1],reverse=True)]
                #candiUnder = under[idx]
                
                for destination in candiUnder:
                    rescpu = CPU_t[destination]+cpu_t[m]
                    resmem = MEM_t[destination]+mem_t[m]
                    if destination != s and \
                        rescpu < nodethis.cpu_capacity and \
                        resmem < nodethis.mem_capacity : # 如果要迁
                        CPU_t[s] -= cpu_t[m]
                        CPU_t[destination] = rescpu
                        MEM_t[s] -=mem_t[m]
                        MEM_t[destination] = resmem
                        if m not in candidate:
                            candidate[m]=[(s,destination)]
                        else:
                            candidate[m].append((s,destination))
                        break
                
        if len(candidate) > 0:
            for k in nodes.keys():
                if nodes[k].cpu_capacity < CPU_t[k] or nodes[k].mem_capacity <  MEM_t[k] :
                    return None,None,False
        if len(candidate) <= 0:
            return None,None,False
        return CPU_t,MEM_t,True
##########################################################################

    def CostOfnodeLoadBalanceSimplify(self,t,b,N,M):
        x_t = np.zeros(shape=(M, N))
        cluster = self.cluster
        for macid,v in cluster.nodes.items():
            incids = [ inc_id for inc_id in v.containers.keys()]
            x_t[macid][incids] = 1
        #print("\tx_t=",x_t)
        ins_ids = cluster.containers.keys()
        Vm = {}
        for vmid in ins_ids:
            pms = np.where(x_t[:,vmid]==1)[0]
            if len(pms)!=1:
                Vm[vmid]=pms
        print("Vm:",Vm)
        vm_cpu = cluster.vm_cpu
        vm_mem = cluster.vm_mem
        cpu_t = vm_cpu[:,t]
        mem_t = vm_mem[:,t]
        cost_bal = 0
        #bal_array = np.zeros(x_t.shape[1])
        for pm in range(x_t.shape[0]): # 对每一列，即每台机器
            vm = np.where(x_t[pm, :] == 1)[0] # 在该机器上的VM索引
            cpu = cpu_t[vm] # 对应VM的cpu
            mem = mem_t[vm]
            #pm_cost = 0
            for i in range(len(vm)-1): # 每台VM与其他VM两两相乘
                c = cpu[i] * np.sum(cpu[i+1:])
                m = mem[i] * np.sum(mem[i+1:])
                cost_bal += c + b * m # cpu乘积+b*mem乘积，所有机器之和
                #pm_cost += c + b * m
            #bal_array[pm] = pm_cost
        
        return cost_bal
