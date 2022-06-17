python -m torch.distributed.launch --master_port 11455 --nproc_per_node=2 \
        main_tdn.py  somethingv1  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
        --lr_scheduler step --lr_steps 30 45 55 --epochs 60 --batch-size 4 \
        --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 \
        --npb