export PYTHONPATH="$PYTHONPATH:$(pwd)"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt /root/shared-nvme/WaveMamba/options/train_mamba.yml
# CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt /root/shared-nvme/Wave-Mamba-main/options/train_wavemamba_lolv2_real.yml
# CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt /root/shared-nvme/Wave-Mamba-main/options/train_wavemamba_lolv2_syn.yml
