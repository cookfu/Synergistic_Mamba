from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
from .cal_ssim import SSIM
from torch import nn
import sys
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)
    
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

@MODEL_REGISTRY.register()
class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.ssim = SSIM().cuda()
        self.dwt = DWT()
        self.iwt = IWT()
        self.l1 = nn.L1Loss().cuda()

        # 敌营评价指标函数
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
                print(self.metric_funcs[name])


        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # print('#########################################################################',load_path)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            # self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('mse_opt'):
            self.cri_mse = build_loss(train_opt['mse_opt']).to(self.device)
        else:
            self.cri_mse = None

        if train_opt.get('lpips_opt'):
            self.cri_lpips = build_loss(train_opt['lpips_opt']).to(self.device)
        else:
            self.cri_lpips = None

        if train_opt.get('char_opt'):
            self.cri_char = build_loss(train_opt['char_opt']).to(self.device)
        else:
            self.cri_char = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if train_opt.get('vgg_opt'):
            self.cri_vgg = build_loss(train_opt['vgg_opt']).to(self.device)
        else:
            self.cri_vgg = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('wave_high_opt'):
            self.cri_wave_high = build_loss(train_opt['wave_high_opt']).to(self.device)
        else:
            self.cri_wave_high = None

        if train_opt.get('wave_low_opt'):
            self.cri_wave_low = build_loss(train_opt['wave_low_opt']).to(self.device)
        else:
            self.cri_wave_low = None
        if train_opt.get('muti_vgg_opt'):
            self.cri_muti_vgg = build_loss(train_opt['muti_vgg_opt']).to(self.device)
        else:
            self.cri_muti_vgg = None

        
        if train_opt.get('prior_opt'):
            self.cri_prior = build_loss(train_opt['prior_opt']).to(self.device)
        else:
            self.cri_prior = None
        if train_opt.get('raw_opt'):
            self.cri_raw = build_loss(train_opt['raw_opt']).to(self.device)
        else:
            self.cri_raw = None

        

        

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # define optimizer
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # # optimizer d
        # optim_type = train_opt['optim_d'].pop('type')
        # optim_class = getattr(torch.optim, optim_type)
        # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        # self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        # for p in self.net_d.parameters():
        #     p.requires_grad = False
        self.optimizer_g.zero_grad()

        self.raw_out,self.output = self.net_g(self.lq)

        # if current_iter==0:

        l_g_total = 0
        loss_dict = OrderedDict()

        # l_pix = self.l1(self.output, self.gt)
        l_pix = self.cri_pix(self.output, self.gt)
        # l_pix = 1 - self.ssim(self.output, self.gt)
        l_g_total += l_pix
        loss_dict['l_pix'] = l_pix

        if train_opt.get('raw_opt', None):
            l_raw = self.cri_raw(self.raw_out, self.gt)
            l_g_total += l_raw
            loss_dict['l_raw'] = l_raw

        if train_opt.get('fft_opt', None):
            l_fft = self.cri_fft(self.output, self.gt)
            l_g_total += l_fft
            loss_dict['l_freq'] = l_fft

        if train_opt.get('prior_opt', None):
            l_prior = self.cri_prior(self.x_ori, self.gt)
            l_g_total += l_prior
            loss_dict['l_prior'] = l_prior

        if train_opt.get('mse_opt', None):
            l_mse = self.cri_mse(self.output, self.gt)
            l_g_total += l_mse
            loss_dict['l_mse'] = l_mse

        if train_opt.get('lpips_opt', None):
            l_lpips,_ = self.cri_lpips(self.output, self.gt)
            l_g_total += l_lpips
            loss_dict['l_lpips'] = l_lpips

        if train_opt.get('char_opt', None):
            l_char = self.cri_char(self.output, self.gt)
            l_g_total += l_char
            loss_dict['l_char'] = l_char

        if train_opt.get('color_opt', None):
            l_color = self.cri_color(self.output, self.gt)
            l_g_total += l_color
            loss_dict['l_color'] = l_color

        if train_opt.get('vgg_opt', None):
            l_vgg = self.cri_vgg(self.output, self.gt)
            l_g_total += l_vgg
            loss_dict['l_vgg'] = l_vgg

        if train_opt.get('ssim_opt', None):
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_g_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        if train_opt.get('wave_high_opt', None):
            self.x_LL, self.x_HL, self.x_LH, self.x_HH = self.dwt(self.gt)
            # l_wave_low = self.cri_wave(self.low_output, x_LL)
            l_wave_high = self.cri_wave_high(self.high_output, torch.cat([self.x_HL, self.x_LH, self.x_HH], dim=1))
            # l_wave = l_wave_low + l_wave_high
            l_g_total += l_wave_high
            loss_dict['l_wave_high'] = l_wave_high

        if train_opt.get('wave_low_opt', None):
            # x_LL, x_HL, x_LH, x_HH = self.dwt(self.gt)
            # l_wave_low = self.cri_wave(self.low_output, x_LL)
            l_wave_low = self.cri_wave_low(self.low_output, self.x_LL)
            # l_wave = l_wave_low + l_wave_high
            l_g_total += l_wave_low
            loss_dict['l_wave_low'] = l_wave_low

        if train_opt.get('muti_vgg_opt', None):
            l_muti_vgg = self.cri_muti_vgg(self.output,self.out_2,self.out_4, self.gt)
            l_g_total += l_muti_vgg
            loss_dict['l_muti_vgg'] = l_muti_vgg

        
        # l_g_total.mean().backward()
        l_g_total.mean().backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        # restoration = self.net_g(self.lq)
        _, _, h, w = lq_input.shape
        if h * w < min_size:
            # out_img, feature_degradation, self.output = self.net_g(self.lq, feature=feature_degradation)
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
                           save_img, save_as_dir):
        # dataset_name = dataloader.dataset.opt['name']
        dataset_name = 'NTIRE2024'
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # print(name)
                    tmp_result = self.metric_funcs[name](*metric_data)
                    
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
                    # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    # self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
#        if tb_logger:
#            for metric, value in self.metric_results.items():
#                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        # self.save_network(self.net_d, 'net_d', current_iter, epoch)
        self.save_training_state(epoch, current_iter)
