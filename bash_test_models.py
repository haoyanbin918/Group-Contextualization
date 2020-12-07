#!/vireo00/yanbin/anaconda3/envs/py37/bin/python
import os

cmd = "/vireo00/yanbin/anaconda3/envs/py37/bin/python test_models.py somethingv2 RGB \
     --model TSN --arch resnet50 --test_nets E33D --test_segments 16 \
     --test_cdivs 16 \
     --batch-size 30 -j 8 --consensus_type=avg \
     --full_res --twice_sample"

os.system(cmd)