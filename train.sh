export PYTHONPATH="$PYTHONPATH:$(pwd)"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train_mamba.yml
