python -m torch.distributed.run --nproc_per_node=2 \
       main.py --config configs/maskformer_coco.yaml