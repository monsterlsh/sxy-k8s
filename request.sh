#!/bin/bash
# set -e
#访问请求
while true
do
	ab -n 20000 -c 100 http://54.160.35.174:31273/getserverinfo | grep $1>/dev/null 2>&1 
	echo "#################################### Request send ####################################"
	sleep 1
done
