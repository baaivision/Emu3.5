source /root/miniconda3/bin/activate
conda activate /root/miniconda3/envs/flagscale-inference 

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345

CUDA_VISIBLE_DEVICES=0,1 python inference.py --cfg configs/example_config_x2i.py
