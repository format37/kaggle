ulimit -n 65536
cd /mnt/storage/share/alex/projects/HandyRL/
CUDA_VISIBLE_DEVICES=0 python3 main.py --train

