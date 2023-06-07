# python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 \
#        train.py --exp_id retrain_stage3 \
#        --stage 3 \
#        --load_network saves/XMem-s0.pth \
#        --use_IS


# python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 \
#        train.py --exp_id finetune_st4_fix_1e-6_rmask \
#        --stage 4 \
#        --s4_lr 1e-6 \
#        --load_network saves/XMem-s012.pth


python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 \
       train.py --exp_id retrain_st2_w_ovis \
       --stage 2 \
       --load_network saves/XMem-s01.pth
