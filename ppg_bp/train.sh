#!/bin/bash
# 5 fold cross validation
python train.py --config config.json

python train.py --config config2.json

python train.py --config config3.json

python train.py --config config4.json

python train.py --config config5.json

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