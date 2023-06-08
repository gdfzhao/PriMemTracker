"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *

# from aggregate import aggregate
# from modules import *
# from memory_util import *



# here put the import lib
from torch import nn
from addict import Dict
from collections import OrderedDict

from model.Mask2Former.modeling.backbone.resnet import ResNet, resnet_spec
from model.Mask2Former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from model.Mask2Former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from model.Mask2Former.modeling.pixel_decoder.ops.modules import MSDeformAttn

class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)
        self._is_train = config.get('train', False)
        self.config = config
        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')
        self.expand_mask = config.get('expand_mask', False)

        # self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object, self.expand_mask, self._is_train)

        if not config['use_IS']:
            self.key_encoder = KeyEncoder()
            # Projection from f16 feature space to key/value space
            self.key_proj = KeyProjection(1024, self.key_dim)
        if not config['share_backbone']:
            self.key_encoder = KeyEncoder()

        self.decoder = Decoder(self.value_dim, self.hidden_dim, self.config)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

        # Enhance key with IS object queries
        if config['use_IS']:
            self.key_conv = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
            self.DeformAttn = MSDeformAttn(n_levels=1)
            self.reference_points = nn.Linear(256, 2)
            self.cat_proj = nn.Conv2d(256 + 100, self.key_dim, kernel_size=1)

            # shrinkage
            self.d_proj = nn.Conv2d(1024, 1, kernel_size=3, padding=1)
            # selection
            self.e_proj = nn.Conv2d(1024, self.key_dim, kernel_size=3, padding=1)

            nn.init.orthogonal_(self.key_conv.weight.data)
            nn.init.zeros_(self.key_conv.bias.data)
            nn.init.orthogonal_(self.cat_proj.weight.data)
            nn.init.zeros_(self.cat_proj.bias.data)

    def encode_key(self, frame, need_sk=True, need_ek=True):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        f16, f8, f4 = self.key_encoder(frame)
        # Similarity 计算中的 选择项 与 收缩项
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4


    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_ * W_, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def enhance_key(self, frame, query_pos, query_src, features, need_sk=True, need_ek=True):

        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        # whether share backbone
        if self.config['share_backbone']:
            f16, f8, f4 = features['res4'], features['res3'], features['res2']
        else:
            f16, f8, f4 = self.key_encoder(frame)
        # ^2 + 1
        shrinkage = self.d_proj(f16) ** 2 + 1 if need_sk else None
        # [0, 1], like attention
        selection = torch.sigmoid(self.e_proj(f16)) if need_ek else None

        Q_g = self.key_conv(f16)     # bs, C_h, H/16, W/16
        (bs, C_h, w, h) = Q_g.shape
        query_src = query_src.transpose(0, 1).contiguous()  # bs, num_queries, C_h
        query_pos = query_pos.transpose(0, 1).contiguous()

        len_q = query_src.shape[1]
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in [Q_g]]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip([Q_g], masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            # pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed = pos_embed
            # lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # spatial_shapes = torch.as_tensor(Q_g.shape[2:], dtype=torch.long, device=Q_g.device)
        # reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        reference_points = self.reference_points(query_src).sigmoid().unsqueeze(2)

        # [bs, 100, 256]
        q_vos = self.DeformAttn(self.with_pos_embed(query_src, query_pos), reference_points, src_flatten,
                              spatial_shapes, level_start_index,
                              mask_flatten)
        # import pdb
        # pdb.set_trace()
        # [bs, 100, 576]
        Q_g_ins = torch.bmm(q_vos, Q_g.flatten(2, 3)).sigmoid().view(bs, len_q, h, w)
        Q_cat = torch.cat((Q_g_ins, Q_g), dim=1)
        enhance_key = self.cat_proj(Q_cat)
        # import pdb
        # pdb.set_trace()
        if need_reshape:
            # B*T*C*H*W
            enhance_key = enhance_key.view(b, t, *enhance_key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()
            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return enhance_key, shrinkage, selection, f16, f8, f4

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True, cls_gt=None):
        num_objects = masks.shape[1]
        # mask.shape:[8, 3, 384, 384] b * num_objects * W * H
        # import ipdb
        # ipdb.set_trace()
        if num_objects != 1:
            # 对每个 num_object 计算其余object 在 mask 上的和，相当于其他目标的前景
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update, cls_gt)

        return g16, h16

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key,
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def segment(self, multi_scale_features, pixel_decoder_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True):
        use_pixel_decoder = self.config.get('use_pixel_decoder', False)

        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out,
                                            pixel_decoder_features=pixel_decoder_features)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'enhance_key':
            return self.enhance_key(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            # import pdb
            # pdb.set_trace()
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0] if 'key_proj.key_proj.weight' in model_weights else 64
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            print(f'Hyperparameters read from the model weights: '
                    f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    if self.expand_mask:
                        pads = torch.zeros((64,2,7,7), device=src_dict[k].device)
                    else:
                        pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
                elif src_dict[k].shape[1] == 5 and self.expand_mask:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict, strict=False)

from addict import Dict

class MaskFormerHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        self.predictor = self.predictor_init(cfg)

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = 4
        transformer_dropout = 0.0
        transformer_nheads = 8
        transformer_dim_feedforward = 1024
        transformer_enc_layers = 4
        conv_dim = 256
        mask_dim = 256
        transformer_in_features = ["res2", "res3", "res4", "res5"] if cfg['use_pixel_decoder'] else ["res3", "res4", "res5"]
        num_feature_levels = 4 if cfg['use_pixel_decoder'] else 3
        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride,
                                                 num_feature_levels)
        return pixel_decoder

    def predictor_init(self, cfg):
        in_channels = 256
        num_classes = 150
        hidden_dim = 256
        num_queries = 100
        nheads = 8
        dim_feedforward = 2048
        dec_layers = 10 - 1
        pre_norm = False
        mask_dim = 256
        enforce_input_project = False
        mask_classification = True
        num_feature_levels = 4 if cfg['use_pixel_decoder'] else 3
        predictor = MultiScaleMaskedTransformerDecoder(in_channels,
                                                       num_classes,
                                                       mask_classification,
                                                       hidden_dim,
                                                       num_queries,
                                                       nheads,
                                                       dim_feedforward,
                                                       dec_layers,
                                                       pre_norm,
                                                       mask_dim,
                                                       enforce_input_project,
                                                       num_feature_levels)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        # import pdb
        # pdb.set_trace()
        predictions, query_pos, query_src = self.predictor(multi_scale_features, mask_features, mask)
        return predictions, query_pos, query_src, multi_scale_features

        # query_pos, query_src = self.predictor(multi_scale_features, mask_features, mask)
        # return query_pos, query_src


class MaskFormerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)

    def build_backbone(self, cfg):
        model_type = 'resnet50'
        assert model_type == 'resnet18' or model_type == 'resnet34' or model_type == 'resnet50', 'Do not support model type!'

        channels = [64, 128, 256, 512]
        if int(model_type[6:]) > 34:
            channels = [item * 4 for item in channels]

        backbone = ResNet(resnet_spec[model_type][0], resnet_spec[model_type][1])
        # backbone.init_weights()
        self.backbone_feature_shape = dict()
        for i, channel in enumerate(channels):
            self.backbone_feature_shape[f'res{i + 2}'] = Dict({'channel': channel, 'stride': 2 ** (i + 2)})
        return backbone

    def load_parallal_model(self, state_dict_):
        state_dict = OrderedDict()
        for key in state_dict_:
            if key.startswith('module') and not key.startswith('module_list'):
                state_dict[key[7:]] = state_dict_[key]
            else:
                state_dict[key] = state_dict_[key]

        # check loaded parameters and created model parameters
        model_state_dict = self.state_dict()
        for key in state_dict:
            if key in model_state_dict:
                if state_dict[key].shape != model_state_dict[key].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        key, model_state_dict[key].shape, state_dict[key].shape))
                    state_dict[key] = model_state_dict[key]
            else:
                print('Drop parameter {}.'.format(key))
        for key in model_state_dict:
            if key not in state_dict:
                print('No param {}.'.format(key))
                state_dict[key] = model_state_dict[key]
        self.load_state_dict(state_dict, strict=False)
        # self.eval()

    def forward(self, inputs):
        # import pdb
        # pdb.set_trace()
        # Determine input shape
        if len(inputs.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = inputs.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            inputs = inputs.flatten(start_dim=0, end_dim=1)
        elif len(inputs.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        features = self.backbone(inputs)
        outputs, query_pos, query_src, pixel_decoder_features = self.sem_seg_head(features)
        # query_pos, query_src = self.sem_seg_head(features)

        # if need_reshape:
        #     # B*C*T*H*W
        #     # outputs = outputs.view(b, t, *outputs.shape[-3:]).transpose(1, 2).contiguous()
        #     if query_pos is not None:
        #         query_pos = query_pos.view(b, t, *query_pos.shape[-3:]).transpose(1, 2).contiguous()
        #     if query_src is not None:
        #         query_src = query_src.view(b, t, *query_src.shape[-3:]).transpose(1, 2).contiguous()

        return outputs, query_pos, query_src, features, pixel_decoder_features
        # return query_pos, query_src

if __name__ == '__main__':
    model = XMem_With_IS()
    input = torch.zeros([8, 3, 480, 480], dtype=torch.float)
    out = model.encode_key(input)
