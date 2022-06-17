python main_tsn.py somethingv1 RGB \
     --arch resnet50 --net M4 --num_segments 8 \
     --element_filter --cdiv 4  --ef_lr5 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --npb
