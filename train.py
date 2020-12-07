#!/vireo00/yanbin/anaconda3/envs/py37/bin/python
import os

# cmd = "/vireo00/yanbin/anaconda3/envs/py37/bin/python main.py somethingv1 RGB \
#      --arch resnet50 --net E33D --num_segments 8 \
#      --element_filter --cdiv 32 \
#      --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
#      --batch-size 20 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#      --npb --full_res --twice_sample"

# os.system(cmd)

# cmd = "/vireo00/yanbin/anaconda3/envs/py37/bin/python main.py somethingv1 RGB \
#      --arch resnet50 --net CLLD --num_segments 8 \
#      --element_filter --cdiv 32 \
#      --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
#      --batch-size 30 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#      --npb --full_res --twice_sample"

# os.system(cmd)

# cmd = "/vireo00/yanbin/anaconda3/envs/py37/bin/python main.py somethingv1 RGB \
#      --arch resnet50 --net T33D --num_segments 8 \
#      --element_filter --cdiv 32 \
#      --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
#      --batch-size 28 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#      --npb --full_res --twice_sample"

# os.system(cmd)

cmd = "/vireo00/yanbin/anaconda3/envs/py37/bin/python main.py somethingv1 RGB \
     --arch resnet50 --net S33D --num_segments 8 \
     --element_filter --cdiv 32 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 26 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --npb --full_res --twice_sample"

os.system(cmd)
