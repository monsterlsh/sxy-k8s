
        

class Container(object):
    def __init__(self,container_config):
        self.id = container_config["id"]
        self.mac_id = container_config["node_id"]
        self.cpu = container_config["cpu"]
        self.mem = container_config["mem"]
        self.memlist = container_config.memory_curve.copy()
        
        self.cpulist = container_config.cpu_curve.copy()
        #加资源
        self.predicts = {}
        self.node = None
        self.lastmacineId = -1
    
    def attach(self, node):
        self.node = node
    
    def predict(self,clock,w,flag=False):
        if flag:
            return self.predicts
        predicts = self.predicts
        if len(self.cpulist)-1>=clock:
            predict_cpu = self.cpulist[clock:clock+w]
            predict_mem = self.memlist[clock:clock+w]
        else :
            predict_cpu = self.cpulist[clock+1:clock+w]
            predict_mem = self.memlist[clock+1:clock+w]
        self.cpu = predict_cpu[0]
        self.mem = predict_mem[0]
        predicts[clock] = {"cpu":predict_cpu,"mem":predict_mem}
        # predicts["cpu"] = predict_cpu
        # predicts["mem"] = predict_mem
        return predicts
