#!/bin/bash
nohup python3 -u ./runk8s.py > ./runk8s.log &
nohup python3 -u ./runk8s.py > ./runk8s-norequest.log &
kubectl get pod -o wide
kubectl top pod | grep tc2 | awk '{print $2}' | tr -cd '[0-9]'