#!/bin/bash
nohup python3 -u ./runk8s.py --algo=sandpiper > ./log/runk8s.log &
nohup python3 -u ./runk8s.py --algo=sxy > ./log/runk8s_sxy.log &
nohup python3 -u ./runk8s.py > ./runk8s-norequest.log &
kubectl get pod -o wide
kubectl top pod | grep tc2 | awk '{print $2}' | tr -cd '[0-9]'
kubectl top pod | grep -P '^tc2' | awk '{print $2}' | tr -cd '[0-9]'

#run runk8s-redis
nohup python3 -u ./runk8s-redis.py --algo=sandpiper > ./log/runk8s-redis-sandpiper.log &

nohup python3 -u ./runk8s-redis.py --algo=sxy > ./log/runk8s-redis-sxy.log &