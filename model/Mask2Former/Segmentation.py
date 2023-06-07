from statistics import mode
from fvcore.common.config import CfgNode
import numpy as np
import os
import cv2
import glob
import tqdm
from PIL import Image
from PIL import ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from modeling.MaskFormerModel import MaskFormerModel
from utils.misc import load_parallal_model
from utils.misc import ADEVisualize

# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog

PALETTE = [(191, 162, 208), (0, 0, 255), (0, 255, 0), (255, 0, 0), (106, 0, 228),
           (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
           (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
           (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
           (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
           (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
           (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
           (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
           (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
           (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
           (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
           (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
           (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
           (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
           (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
           (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
           (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
           (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
           (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
           (246, 0, 122)]


class Segmentation():
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_queries = 100
        self.size_divisibility = 32
        self.num_classes = 150
        self.device = torch.device("cuda", cfg.local_rank)        

        # data processing program
        self.padding_constant = 2**5 # resnet 总共下采样5次
        self.test_dir = cfg.TEST.TEST_DIR
        self.output_dir = cfg.TEST.SAVE_DIR
        self.imgMaxSize = cfg.INPUT.CROP.MAX_SIZE
        self.pixel_mean = np.array(cfg.DATASETS.PIXEL_MEAN)
        self.pixel_std = np.array(cfg.DATASETS.PIXEL_STD)
        self.visualize = ADEVisualize()
        self.model = MaskFormerModel(cfg)

        pretrain_weights = cfg.MODEL.PRETRAINED_WEIGHTS
        assert os.path.exists(pretrain_weights), f'please check weights file: {cfg.MODEL.PRETRAINED_WEIGHTS}'
        self.load_model(pretrain_weights)
        
    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')

        ckpt_dict = state_dict['model']

        self.last_lr = state_dict['lr']
        self.start_epoch = state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("loaded pretrain mode:{}".format(pretrain_weights))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.   
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1))
        return img

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

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

    def image_preprocess(self, img):
        img_height, img_width = img.shape[0], img.shape[1]
        this_scale = self.imgMaxSize / max(img_height, img_width)
        target_width = img_width * this_scale
        target_height = img_height * this_scale
        input_width = int(self.round2nearest_multiple(target_width, self.padding_constant))
        input_height = int(self.round2nearest_multiple(target_height, self.padding_constant))

        img, padding_info = self.resize_padding(Image.fromarray(img), (input_width, input_height))
        img = self.img_transform(img)

        transformer_info = {'padding_info': padding_info, 'scale': this_scale, 'input_size':(input_height, input_width)}
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        return input_tensor, transformer_info

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        mask_pred = mask_pred.sigmoid()  
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
        return semseg.cpu().numpy()

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        mask_cls, mask_pred = mask_cls.squeeze(0), mask_pred.squeeze(0)
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, 1:]
        # scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # scores = F.softmax(mask_cls, dim=-1)[..., 1:]
        # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        labels = torch.arange(150, device=self.device).unsqueeze(0).repeat(100, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(2, sorted=False)
        labels_per_image = labels[topk_indices]

        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = topk_indices // 150
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                    pred_masks.flatten(1).sum(1) + 1e-6)
        scores = scores_per_image * mask_scores_per_image
        return pred_masks, scores, labels_per_image
        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
        #     keep = torch.zeros_like(scores_per_image).bool()
        #     for i, lab in enumerate(labels_per_image):
        #         keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
        #
        #     scores_per_image = scores_per_image[keep]
        #     labels_per_image = labels_per_image[keep]
        #     mask_pred = mask_pred[keep]
        #
        # result = Instances(image_size)
        # # mask (before sigmoid)
        # result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # # Uncomment the following to get boxes from masks (this is slow)
        # # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        #
        # # calculate average mask prob
        # mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        # result.scores = scores_per_image * mask_scores_per_image
        # result.pred_classes = labels_per_image
        # return result

    def postprocess(self, pred_mask, transformer_info, target_size):       
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        padding_info = transformer_info['padding_info'] 
        
        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        mask = pred_mask[top: oh - bottom, left: ow - right]
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    @torch.no_grad()
    def forward(self, img_list=None):
        # ade20k_metadata = MetadataCatalog.get("ade20k_sem_seg_val")
        if img_list is None or len(img_list) == 0:
            img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
        
        for image_path in tqdm.tqdm(img_list):
            img_name = os.path.basename(image_path)
            seg_name = img_name.split('.')[0] + '_seg.png'
            output_path = os.path.join(self.output_dir, seg_name)
            img = Image.open(image_path).convert('RGB')
            img_height, img_width = img.size[1], img.size[0]
            inpurt_tensor, transformer_info = self.image_preprocess(np.array(img))       

            outputs, query_pos, query_src = self.model(inpurt_tensor)
            import pdb
            pdb.set_trace()
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(inpurt_tensor.shape[-2], inpurt_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            # pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
            pred_masks, scores, labels_per_image = self.instance_inference(mask_cls_results, mask_pred_results)
            # instance
            # [1, num_query, H, W]
            # import pdb
            # pdb.set_trace()
            color_mask = np.zeros((pred_masks.shape[1], pred_masks.shape[2], 3))
            pred_masks = pred_masks.float().cpu().numpy()
            for instance_id in range(len(labels_per_image.cpu())):
                ci = int(labels_per_image[instance_id]) % len(PALETTE)
                color_mask[:, :, 0][pred_masks[instance_id] == 1] = PALETTE[ci][0]
                color_mask[:, :, 1][pred_masks[instance_id] == 1] = PALETTE[ci][1]
                color_mask[:, :, 2][pred_masks[instance_id] == 1] = PALETTE[ci][2]

            color_mask = self.postprocess(color_mask, transformer_info, (img_width, img_height))
            ori_img = cv2.imread(
                os.path.join(image_path))
            # color_mask = ori_img + color_mask * 0.5
            cv2.imwrite(output_path, color_mask)


            # mask_img = np.zeros((pred_masks.shape[1:]))
            # for i in range(pred_masks.shape[0]):
            #     mask_img += (i + 1) * (pred_masks[i, :, :] > 0)

            # # import pdb
            # # pdb.set_trace()
            # self.visualize.show_result(img, mask_img, output_path)


            # semantic
            # mask_img = np.argmax(pred_masks, axis=1)[0]
            # mask_img = self.postprocess(mask_img, transformer_info, (img_width, img_height))
            # self.visualize.show_result(img, mask_img, output_path)


            # v = Visualizer(np.array(img), ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            # semantic_result = v.draw_sem_seg(mask_img).get_image()
            # cv2.imwrite(output_path, semantic_result)