import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def create_val_dataloader(opt, logger): #创建训练和测试的dataloader
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            #创建测试数据的dataset和dataloader
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')

    return val_loader

def init_tb_loggers(opt): #初始化logger函数
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def test_pipeline(root_path):
    # 解析外部参数，设置分布式设置，设置随机种子
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 创建文件夹并且初始化logger
    # print(opt)
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = init_tb_loggers(opt)

    val_loader = create_val_dataloader(opt, logger)


    # 创建模型
    model = build_model(opt)

    if len(val_loaders) > 1:
            logger.warning('Multiple validation datasets are *only* supported by SRModel.')
    # for val_loader in val_loaders:
    model.validation(val_loader, current_iter,epoch, tb_logger, opt['val']['save_img'])


if __name__ == '__main__':
    # root_path = '/root/shared-nvme/Wave-Mamba-main/options/test_wavemamba_lolv2_syn.yml'
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
