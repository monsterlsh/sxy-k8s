#!/bin/bash

for podid in {0..3}
do
    
    podname="tc""$podid"
    nodeid=`expr ${podid} % 2 + 1`
    nodename="k8s-node${nodeid}"
    echo "podname = ${podname} nodename=${nodename}"
    kubectl delete pod ${podname}
    sed -i "4c\  name: ${podname}"  /root/tomcat/pod.yaml
    sed -i "8c\  nodeName: ${nodename}"  /root/tomcat/pod.yaml
    kubectl create -f /root/tomcat/pod.yaml
done
