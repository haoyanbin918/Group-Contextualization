# test a single checkpoint
python test_models_tsntsmgst_gc.py somethingv1 RGB \
     --arch resnet50 --test_nets M4 --test_segments 8 \
     --element_filter --cdiv 4 \
     --batch-size 20 -j 4 --consensus_type=avg \
     --twice_sample --full_res \
     --test_weights=path-to-your-checkpoint/ckpt.best.pth.tar

# test two checkpoints, ensemble

python test_models_tsntsmgst_gc.py somethingv1 RGB \
     --arch resnet50 --test_nets M4,M4 --test_segments 8,16 \
     --element_filter --cdiv 4 \
     --batch-size 20 -j 4 --consensus_type=avg \
     --twice_sample --full_res \
     --test_weights=path1-to-your-checkpoint/ckpt_test.best.pth.tar,path2-to-your-checkpoint/ckpt_test.best.pth.tar
