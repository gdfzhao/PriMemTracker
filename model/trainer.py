"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torch.nn import functional as F
import cv2

from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from model.network import MaskFormerModel
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class XMemTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        self.XMem = nn.parallel.DistributedDataParallel(
            XMem(config).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        if config['use_IS']:
            self.Mask2Former = nn.parallel.DistributedDataParallel(
                MaskFormerModel(config).cuda(),
                device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)


        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.XMem.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        # print(filter(lambda p: p.requires_grad, self.Mask2Former.parameters()))
        if not config['use_IS']:
            self.optimizer = optim.AdamW(filter(
                lambda p: p.requires_grad, self.XMem.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = optim.AdamW(
                [{'params': filter(lambda p: p.requires_grad, self.XMem.parameters()), 'lr': config['lr']},
                 {'params': filter(lambda p: p.requires_grad, self.Mask2Former.parameters()), 'lr': config['lr'] / 10}],
                weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1
        if config['use_IS']:
            self.pixel_mean = np.array([0.485, 0.456, 0.406])
            self.pixel_std = np.array([0.229, 0.224, 0.225])
            self.device = torch.device("cuda", self.local_rank)
            self.mapper = MaskMapper()
    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Interpolation)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img, [left, top, right, bottom]

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1))
        return img

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def image_preprocess(self, img):
        img_height, img_width = img.shape[0], img.shape[1]
        this_scale = 512 / max(img_height, img_width)
        target_width = img_width * this_scale
        target_height = img_height * this_scale
        input_width = int(self.round2nearest_multiple(target_width, 2**5))
        input_height = int(self.round2nearest_multiple(target_height, 2**5))

        img, padding_info = self.resize_padding(Image.fromarray(img), (input_width, input_height))
        img = self.img_transform(img)

        transformer_info = {'padding_info': padding_info, 'scale': this_scale,
                            'input_size': (input_height, input_width)}
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(torch.device("cuda", self.local_rank))
        return input_tensor, transformer_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        scores = F.softmax(mask_cls, dim=-1)[..., 1:]
        # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        labels = torch.arange(150, device=self.device).unsqueeze(0).repeat(100, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(1, sorted=False)
        labels_per_image = labels[topk_indices]

        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = topk_indices // 150
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        # mask_pred = mask_pred[topk_indices]
        pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6)
        scores = scores_per_image * mask_scores_per_image
        return scores

    def postprocess(self, pred_mask, transformer_info, target_size):
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        padding_info = transformer_info['padding_info']

        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        mask = pred_mask[top: oh - bottom, left: ow - right]
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            # TODO: add first frame mask embedding

            if not self.config['use_IS']:
                key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)
                # print(key.shape) # [bs, 64, 8, 24, 24]
                # import pdb
                # pdb.set_trace()

            # --------------------- * ----------------------
            if self.config['use_IS']:
                outputs, query_pos, query_src, features, pixel_decoder_features = self.Mask2Former(frames)
                key, shrinkage, selection, f16, f8, f4 = self.XMem('enhance_key', frames, query_pos, query_src, features)  # [bs, 64, 8, 24, 24]

                if self.config['use_pixel_decoder']:
                    b, t = f16.shape[:2]
                    pixel_decoder_features = [pf.view(b, t, *pf.shape[-3:]) for pf in pixel_decoder_features]
                else:
                    pixel_decoder_features = None
                # import pdb
                # pdb.set_trace()
                # ori_images = data['info']['ori_images']
                # ori_image_tensors = []
                # trs_infos = []
                # for i in range(len(ori_images)):
                #     for image_path in ori_images[i]:
                #         img = Image.open(image_path).convert('RGB')
                #         img_height, img_width = img.size[1], img.size[0]
                #         input_tensor, transformer_info = self.image_preprocess(np.array(img))
                #         ori_image_tensors.append(input_tensor)
                #         trs_infos.append(transformer_info)
                #         outputs, query_pos, query_src = self.Mask2Former(input_tensor)
                #
                #         out = self.XMem('enhance_key', f16[:, 0], query_pos, query_src)
                        # mask_cls_results = outputs['pred_logits']
                        # mask_pred_results = outputs['pred_masks']


            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            # 只对序列中的第一帧进行encode value
            # print(data['cls_gt_one_hot'].shape)
            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0], is_deep_update=True, cls_gt=data['cls_gt_one_hot'][:,0])
            values = v16.unsqueeze(3) # add the time dimension

            for ti in range(1, self.num_frames):
                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would 
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                # Segment frame ti
                memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), [p[:,ti] for p in pixel_decoder_features] if self.config['use_pixel_decoder'] else None,
                                                  memory_readout, hidden, selector, h_out=(ti < (self.num_frames-1)))
                # import pdb
                # pdb.set_trace()

                # 前一帧的预测结果传递
                # No need to encode the last frame
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update,
                                            cls_gt=data['cls_gt_one_hot'][:,ti])
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits


            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()

        self.scheduler.step()

    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.XMem.module.state_dict(), model_path)
        if self.config['use_IS']:
            maskformer_model_path = f'{self.save_path}_mask2former_{it}.pth'
            torch.save(self.Mask2Former.module.state_dict(), maskformer_model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if not self.config['use_IS']:
            checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
            checkpoint = {
                'it': it,
                'network': self.XMem.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint_path = f'{self.save_path}_checkpoint_with_IS_{it}.pth'
            checkpoint = {
                'it': it,
                'network': {'XMem': self.XMem.module.state_dict(), 'Mask2Former': self.Mask2Former.module.state_dict()},
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        if not self.config['use_IS']:
            self.XMem.module.load_state_dict(network)
        else:
            self.XMem.module.load_state_dict(network['XMem'])
            self.Mask2Former.module.load_state_dict(network['Mask2Former'])
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict, IS_src_dict=None):

        self.XMem.module.load_weights(src_dict)
        if self.config['use_IS']:
            self.Mask2Former.module.load_parallal_model(IS_src_dict['model'])
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})
        IS_src_dict = None
        if self.config['use_IS']:
            # IS_src_dict = torch.load("model/Mask2Former/ckpt/mask2former_resnet50.pth",
            #                          map_location=map_location)
            IS_src_dict = torch.load("/dfs/data/VOS/XMem/model/Mask2Former/ckpt/mask2former_Epoch40_dice0.6557618501400757.pth",
                                     map_location=map_location)
        self.load_network_in_memory(src_dict, IS_src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.XMem.eval()
        # self.XMem.value_encoder._is_train = True
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self

