from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from algorithm import Scheduler
from time import sleep,time
from sys import exit
import csv
from operator import eq
class ScheduleSys:
    def __init__(self,nodeNum,podNum,algo) -> None:
        self.cpudict = {} #podname:cpulist
        self.memdict = {} #podname:memlist
        self.nodeNum = nodeNum
        self.podNum = podNum
        self.algorithm = Scheduler()
        self.algoName = algo
        self.nodes = {} #所有nodename：set(podname)
        self.pods = set() #所有podname
        self.startTime = time()
        
        pass
    
    def schedule(self):
        """
        1.每隔1小时获取资源量cpu mem
        2.调用算法
        """
        t = 0
        podnum = self.podNum
        #os.system("bash ./request.sh")
        flag = self.checkNodeAndPodCmd()
        while flag:
            print(f"################################# {t}th loading #############################")
            for i in range(podnum):
                podname = "tc"+str(i)
                self.getCpuMemNow(podname,t)
            
            self.NodeToPod(t,self.algoName)
            #print(f"pods is {self.pods}\nfirst nodes= \n{self.nodes}")
            #主要现在这个sandpiper函数里的一些细节 日志文件在run8s。log
            if t==0:
                Filename = './metric/sandpiper.csv' if algo == "sandpiper" else './metric/sxy.csv'
                with open( Filename,'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["value","eval_bal","eval_mig"])
            if self.algoName == "sandpiper":
                newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName) #调度算法后生成新的node pod分配方案
            elif self.algoName == "sxy":
                # if t>20:
                #     newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName,self.cluster,t)
                # else:
                #     newnodes = self.nodes
                newnodes = self.algorithm(self.nodes,self.cpudict,self.memdict,self.algoName,self.cluster,t)
            else:
                print("请选择调度算法 --algo=sandpiper or --algo=sxy\n结束")
                exit()
            #判断有无迁移
            if  self.nodes == newnodes or eq(self.nodes,newnodes):
                
                sleep(5)
                flag = self.checkNodeAndPodCmd()
                t=t+1
                # if t>5:
                #     break;
                continue
            self.nodes = newnodes
            print(f"after schedule nodes= \n {self.nodes}")
            
            assert len(self.nodes)==self.nodeNum and len(self.pods) == self.podNum
            self.modifyYaml() #迁移
            sleep(5)
            flag = self.checkNodeAndPodCmd()
            
            
            """
            # 下面的都是test是吗？实际上应该是sleep(3600)一句？
            #test
            t = t+1
            if t > 0:
                break
            #test

            # 实际测试
            
            """
            sleep(5)
            t=t+1

            if t>5:
                print(f"at {t} Done")
                break
    
    def NodeToPod(self,t,algoName):
        """第一时刻初始化 node和pod的对应关系
            问题：如果 将pods nodes作为返回传入self.pods，self.nodes，会出现部分pod无法分配到node里
        """
        if t == 0 and algoName == "sxy":
            isSxy  = True
        else:
            isSxy = False
        if isSxy:
            # TODO clusster node container
            from sxyAlgo.cluster import Cluster
            self.cluster = Cluster()
            cluster = self.cluster
            from sxyAlgo.container import Container
            from sxyAlgo.node import Node
        pods = self.pods
        nodes = self.nodes
        for podname in self.cpudict.keys():
            with os.popen("kubectl get pod -o wide|grep "+podname+ " | awk '{print $9}'") as p :
                pods.add(podname)
                nodename = p.read()[0:-1] #k8s-node1 k8s-node2这样的 自定义的
            if nodename == "<none>":
                with os.popen("kubectl get pod -o wide|grep "+podname+ " | awk '{print $7}'") as p :
                    nodename = p.read()[0:-1]
            if nodename not in self.nodes:
                nodes[nodename]=set()
                if isSxy:
                    node_id = int(nodename[-1:])-1
                    node_config = {"nodeName":nodename,"id":node_id,"cpu_capacity":100,"mem_capacity":100}
                    node = Node(node_config)
                    cluster.nodes[node.id] = node
                    node.attach(self)
            pod_id = int(podname[-1:])
            cpu = self.cpudict[podname][-1]
            mem = self.memdict[podname][-1]
            if isSxy:
                container_config = {"containerName":podname,"id":pod_id,"node_id":node_id,\
                    "cpu":cpu,"mem":mem,\
                        "memory_curve":[],"cpu_curve":[]}
                container = Container(container_config)
                cluster.containers[container.id] = container
                node_id = container.mac_id
                node = cluster.nodes.get(node_id, None)
                assert node is not None
                node.push(container)
            if algoName == "sxy":
                    container = self.cluster.containers[pod_id]
                    container.memlist.append(mem)
                    container.cpulist.append(cpu)
            #if t==0:
            nodes[nodename].add(podname)
            
        

    def getCpuMemNow(self,podname,t=0):
        """获取当前时刻的资源信息"""
        cpudict  = self.cpudict
        memdict = self.memdict
        if podname not in cpudict:
            cpudict[podname] = []
        if podname not in memdict:
            memdict[podname] = []
        #kubectl top pod | grep tc0 | awk '{print $3}' | tr -cd '[0-9]'
        cpu,mem = '',''
        looptimes = 0
        while cpu=='' or mem=='':
            print(f"podname={podname} In getCpuMemNow spending cpu={cpu},mem={mem}")
            sleep(3)
            looptimes += 1
            if looptimes > 5:
                with os.popen("kubectl top pod | grep "+podname) as p:
                    print("wrong",p.read())
                exit(1)
            with os.popen("kubectl top pod | grep "+podname+" | awk '{print $2}' | tr -cd '[0-9]'") as cmdcpu :
                
                cpu = cmdcpu.read()
            with os.popen("kubectl top pod | grep "+podname+" | awk '{print $3}' | tr -cd '[0-9]'") as cmdmem :
                
                mem = cmdmem.read()
        print(f"podname={podname} cpu: {cpu} mem:{mem}")
        intcpu = int(cpu)
        intmem = int(mem)
        # try:
        #     intcpu = int(cpu) # 转为int
        # except:
        #     cmdcpu = os.popen("kubectl top pod | grep "+podname+" | awk '{print $2}' | tr -cd '[0-9]'")
        #     intcpu = int(cpu)
        # try:
        #     intmem = int(cmdmem.read())
        # except:
        #     cmdmem = os.popen("kubectl top pod | grep "+podname+" | awk '{print $3}' | tr -cd '[0-9]'")
        #     intmem = int(cmdmem.read())
            
        # 计算比例
        cpuperc = intcpu / 20 # 即/2000*100
        memperc = intmem / 2400 # 即/240000*100
        if t == 0 :
            self.cpudict[podname] = [0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05]
            self.memdict[podname] = [0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05]
        self.cpudict[podname].append(cpuperc)
        self.memdict[podname].append(memperc)
        # print(intcpu)
        # assert 1==0
    # 驱逐pod-修改yaml文件-重建pod
    def modifyYaml(self):
        #migrated_pod_list = self.pods # 需要迁移的pod
        print("migration start")
        nodes = self.nodes
        p =1
        for node_name,migrated_pod_list in nodes.items():
            for pod_name in migrated_pod_list:
                if p==1:
                    p =0
                    print("modify ",pod_name )
                    # po = os.popen("kubectl delete pod "+pod_name+"& ""sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml"\
                    #     "& sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml"\
                    #         "& kubectl create -f /root/tomcat/pod.yaml") # 驱逐pod_name
                    with os.popen("kubectl delete pod "+pod_name) as po:
                        print(1,po.read())
                        
                    with os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") as po:
                        print(2,po.read())
                        #sleep(0.1)
                    
                    with os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") as po:
                        print(3,po.read())
                        #sleep(0.1)
                    
                    with os.popen("kubectl create -f /root/tomcat/pod.yaml") as po:
                        print(4,po.read())
                        #sleep(0.1)
                    
                if po !=None:
                    p = 1
        
                # os.popen("sed -i '4c\  name: "+pod_name+"' /root/tomcat/pod.yaml") # 改为pod_name
                # os.popen("sed -i '8c\  nodeName: "+node_name+"' /root/tomcat/pod.yaml") # 改为node_name
                # os.popen("kubectl create -f /root/tomcat/pod.yaml")
        #sleep(5)
    def checkNodeAndPodCmd(self):

        cmd = os.popen("kubectl get pod -o wide")
        
        print(f"\nat time {time()-self.startTime}: \n{cmd.read()} \n")
        print(self.cpudict)
        return True

if __name__ =="__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--algo",type=str)
    args = parse.parse_args()
    algo = args.algo
    test = ScheduleSys(2,4,algo)
    test.schedule()
    
