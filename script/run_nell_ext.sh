#!/bin/sh

kge='TransE'
gpu=0

python main.py --data_path ./data/nell_ext.pkl --task_name nell_ext_transe \
        --kge ${kge} --gpu cuda:${gpu}