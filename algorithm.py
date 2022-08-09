
import numpy as np
import time
from sandpiper import Sandpiper_algo
from abc import ABC,abstractmethod
from sxyAlgo.Algorithm_sxy import Algorithm_sxy
from utl import CostOfLoadBalance,CostOfMigration
import csv
class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass

class Scheduler():
    def __init__(self) -> None:
        self.sxy_algo = Algorithm_sxy()
        self.Filename = None
        pass
    def __call__(self,nodes,cpudict,memdict,algo,cluster=None,t=None):
        #nodes,cpudict,memdict = self.nodes,self.cpudict,self.memdict
        podnum = len(cpudict)
        nodenum = len(nodes)
        print("podnum = {} nodenum = {}".format(podnum,nodenum))
        if algo == "sandpiper":
            self.x_t0 = np.zeros([podnum,nodenum]) #初始化矩阵
            self.cpu_t0, self.mem_t0 = np.zeros(podnum),np.zeros(podnum) #初始化 cpu和mem
            self.getMatrix(nodes,cpudict,memdict)
            placement = Sandpiper_algo(self.x_t0, self.cpu_t0, self.mem_t0,CPU_MAX=30,MEM_MAX=30)
            eval_bal = CostOfLoadBalance(self.cpu_t0,self.mem_t0 , placement,b=0.0025)
            eval_mig = CostOfMigration(self.x_t0, placement,self.mem_t0)
            value = eval_bal+(podnum-1)*eval_mig*0.004
        elif algo == "sxy":
            #TODO sxy algorithm
            print("sxy")
            value,eval_bal,eval_mig = self.sxy_algo(cluster,t)
        #print(f"placement = {placement}")
        if self.Filename == None:
            self.Filename = './metric/sandpiper.csv' if algo == "sandpiper" else './metric/sxy.csv'
        with open( self.Filename,'a') as f:
            writer = csv.writer(f)
            writer.writerow([value,eval_bal,eval_mig])
        if algo == "sxy":
            return self.sxyDict(cluster)
        return self.MatrixToDict(placement,nodenum)
    def sxyDict(self,cluster):
        nodes = cluster.nodes
        newnodes = {}
        nodename_prefix = "k8s-node"
        for nodeNum,v in nodes.items():
            newnodes[nodename_prefix+str(nodeNum+1)] = set(["tc"+str(e) for e in v.containers.keys() ])
        return newnodes
    def MatrixToDict(self,placement,nodenum):
        """将placement pod -> node 关系转换为 map； nodename:{tc0,tc1,...}

        Args:
            placement (numpy.ndarray): node和pod的位置关系 N*M (N podnum; M nodenum)
            nodenum (int): node 数量

        Returns:
            dict: nodename:{tc0,tc1,...}
        """
        nodes = {}
        nodename_prefix = "k8s-node"
        for nodeNum in range(1,nodenum+1):
            x = np.where(placement[:,nodeNum-1]==1)[0]
            #print(f"node_{nodeNum} has pod_({x}) and line={placement[:,nodeNum-1]} ")
            nodes[nodename_prefix+str(nodeNum)] = set(["tc"+str(e) for e in x])
        return nodes
    
    def getMatrix(self,nodes:dict,cpudict:dict,memdict:dict):
        """将nodes转换为 二维数组 node和pod的位置关系 N*M (N podnum; M nodenum)

        Args:
            nodes (dict): nodename:{tc0,tc1,...} 
            cpudict (dict): 每个pod的cpu时间序列
            memdict (dict): 每个pod的mem时间序列
        """
        #TODO 这里要把node 和pod 名称转换成数字，我以为 node1 node2  然后pod是 tc0 tc1 tc2
        # nodename:k8s-node1 podname:tc1
        x_t0 = self.x_t0
        for nodename,podnameSet in nodes.items():
            #print("nodename = {} podnameSet = {} x_t0.shape={}".format(nodename,podnameSet,x_t0.shape))
            i = int(nodename[8:])-1 #获取node编号
            for podname in podnameSet:
                j = int(podname[2:])
                x_t0[j][i] = 1
        cpu_t0, mem_t0 = self.cpu_t0, self.mem_t0
        for podenameCpu,podenameMem in zip(cpudict.keys(),memdict.keys()):
            cpu_t0[int(podenameCpu[2:])] = cpudict[podenameCpu][-1]
            mem_t0[int(podenameMem[2:])] = memdict[podenameMem][-1]
    # def Sandpiper(self,x_t0, cpu_t0, mem_t0,CPU_MAX, MEM_MAX):
    #     # 计算当前各PM和VM的Vol值及VSR值
    #     CPU_t0 = ResourceUsage(cpu_t0, x_t0)
    #     MEM_t0 = ResourceUsage(mem_t0, x_t0)
    #     Vol_pm = 10000 / ((100 - CPU_t0) * (100 - MEM_t0)) # 注意每PM/VM的资源需求要<100%
    #     Vol_vm = 10000 / ((100 - cpu_t0) * (100 - mem_t0)) # 1*N矩阵
    #     VSR = Vol_vm/ mem_t0 # 1*N矩阵
    #     # 机器按Vol值按序排序，存储机器号
    #     pm_asc = Vol_pm.argsort()
    #     pm_desc = pm_asc[::-1]
    #     #print('pm_desc',pm_desc)
    #     placement = x_t0.copy() # 初始化
    #     # 按序对每台机器做迁出
    #     pm_asc.astype(int)
    #     pm_desc.astype(int)
    #     migs_inc_outToin = {}
    #     pmout = {}
    #     pmin={}
    #     #print(f'\t to schedule pm num is {len(pm_desc)}')
    #     idx=0
    #     metric_pmout = set()
    #     metric_pmin = set()
    #     for pm_outs in pm_desc:
    #         pm_out=int(pm_outs)
    #         #print('\t\t sand piper',idx)
    #         idx+=1
    #         if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载
    #             continue
    #         # 将每台机器上的VM降序排序，存储VM号
    #         vm_in_pm = np.where(x_t0[:, pm_out] == 1)[0] # 该机器上VM的VM号
    #         VSR_in_pm = VSR[vm_in_pm] # 这些VM的VSR
    #         #print(vm_in_pm,VSR_in_pm)
    #         vm_VSR = np.array([vm_in_pm, VSR_in_pm]) # 二维数组，第一行为VM号，第二行为VM对应VSR值
    #         vm_asc = vm_VSR.T[np.lexsort(vm_VSR)].T # 按照VSR升序排序
    #         #vm_asc = vm_VSR[:,vm_VSR[1].argsort()]
    #         vm_desc = vm_asc[0, ::-1] # 获取降序排序后的VM号
    #         vm_desc.astype(int)
    #         for vms in vm_desc: # 从VSR最大的VM开始被迁移
    #             vm = int(vms)
    #             if CPU_t0[pm_out] <= CPU_MAX and MEM_t0[pm_out] <= MEM_MAX: # 按序迁出该机器上的VM直到机器不过载                
    #                 break
    #             for pm_inx in pm_asc: # 从Vol最小的开始迁入
    #                 pm_in = int(pm_inx)
    #                 if CPU_t0[pm_in] + cpu_t0[vm] <= CPU_MAX and MEM_t0[pm_in] + mem_t0[vm] <= MEM_MAX: # 有机器放得下
    #                     metric_pmout.add(pm_out)
    #                     metric_pmin.add(pm_in)

    #                     placement[vm][pm_out] = 0 # 迁出机器
    #                     CPU_t0[pm_out] = CPU_t0[pm_out] - cpu_t0[vm]
    #                     MEM_t0[pm_out] = MEM_t0[pm_out] - mem_t0[vm]
                        
    #                     placement[vm][pm_in] = 1 # 迁入机器
    #                     CPU_t0[pm_in] = CPU_t0[pm_in] + cpu_t0[vm]
    #                     MEM_t0[pm_in] = MEM_t0[pm_in] + mem_t0[vm]
    #                     if pm_out in pmout:
    #                         pmout[pm_out].append(vm)
    #                     else:
    #                         pmout[pm_out]=[vm]
    #                     if pm_in in pmin:
    #                         pmin[pm_in].append(vm)
    #                     else:
    #                         pmin[pm_in]=[vm]
    #                     migs_inc_outToin[vm]=(pmout,pmin)
    #                     break      
    #         # 循环结束条件是没有机器过载
    #         if (CPU_t0 <= CPU_MAX).all() and (MEM_t0 <= MEM_MAX).all():
    #             break
    #     return np.array(placement)

    