#!/bin/bash
# 5 fold cross validation
python train_fusion.py --config config_face_manual_demo.json

python train_fusion.py --config config_face_manual_demo2.json

python train_fusion.py --config config_face_manual_demo3.json

python train_fusion.py --config config_face_manual_demo4.json

python train_fusion.py --config config_face_manual_demo5.json

python test_fusion.py --config test_config_manual_demo.json
# python train.py --config config_palm.json

# python train.py --config config_palm2.json

# python train.py --config config_palm3.json

# python train.py --config config_palm4.json

# python train.py --config config_palm5.json

# python train.py --config config_face_palm.json

# python train.py --config config_face_palm2.json

# python train.py --config config_face_palm3.json

# python train.py --config config_face_palm4.json

# python train.py --config config_face_palm5.json