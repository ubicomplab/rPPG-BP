#!/bin/bash
# face threshold 0.8 5 fold cross validation
python train_fusion.py --config ./configs/config_face_v2_t8.json

python train_fusion.py --config ./configs/config_face_v2_t8_2.json

python train_fusion.py --config ./configs/config_face_v2_t8_3.json

python train_fusion.py --config ./configs/config_face_v2_t8_4.json

python train_fusion.py --config ./configs/config_face_v2_t8_5.json

# face top 10 5 fold cross validation
python train_fusion.py --config ./configs/config_face_v2_top10_2.json

python train_fusion.py --config ./configs/config_face_v2_t10_2.json

python train_fusion.py --config ./configs/config_face_v2_t10_3.json

python train_fusion.py --config ./configs/config_face_v2_t10_4.json

python train_fusion.py --config ./configs/config_face_v2_t10_5.json

# finger threshold 0.8 5 fold cross validation
python train_fusion.py --config ./configs/config_finger_v2_t8.json

python train_fusion.py --config ./configs/config_finger_v2_t8_2.json

python train_fusion.py --config ./configs/config_finger_v2_t8_3.json

python train_fusion.py --config ./configs/config_finger_v2_t8_4.json

python train_fusion.py --config ./configs/config_finger_v2_t8_5.json

# finger top 10 5 fold cross validation
python train_fusion.py --config ./configs/config_finger_v2_top10_2.json

python train_fusion.py --config ./configs/config_finger_v2_top10_2.json

python train_fusion.py --config ./configs/config_finger_v2_top10_3.json

python train_fusion.py --config ./configs/config_finger_v2_top10_4.json

python train_fusion.py --config ./configs/config_finger_v2_top10_5.json