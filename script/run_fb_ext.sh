#!/bin/sh

kge='TransE'
gpu=0

python main.py --data_path ./data/fb_ext.pkl --task_name fb_ext_transe \
        --kge ${kge} --gpu cuda:${gpu}