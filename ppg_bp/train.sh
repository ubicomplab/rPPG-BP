#!/bin/bash
# 5 fold cross validation
python train_fusion.py --config ./configs/config_face_manual_demo.json

python train_fusion.py --config ./configs/config_face_manual_demo2.json

python train_fusion.py --config ./configs/config_face_manual_demo3.json

python train_fusion.py --config ./configs/config_face_manual_demo4.json

python train_fusion.py --config ./configs/config_face_manual_demo5.json