#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
while true
do
 echo $DIR
 python3 $DIR/main.py $@ 
 
 if [ $? -ne 0 ]; then
  break 
 fi
done
