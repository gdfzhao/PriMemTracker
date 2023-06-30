import os
from os import path
from argparse import ArgumentParser
import shutil

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import vot_py as vot

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset, VOTTestDataset
from inference.data.video_reader import VideoReader
from inference.data.mask_mapper import MaskMapper
from model.network import XMem, MaskFormerModel
from inference.inference_core import InferenceCore

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')
import pickle as pkl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Arguments loading
"""

parser = ArgumentParser()
parser.add_argument('--model', default='/dfs/data/VOS/XMem/saves/retrain_st3_w_ovis_IS_s3/retrain_st3_w_ovis_IS_s3_105000.pth')
# parser.add_argument('--model', default='/dfs/data/VOS/XMem/saves/XMem-s012.pth')
parser.add_argument('--IS_model', help='Path to pretrained network weight only',
                    default="/dfs/data/VOS/XMem/saves/retrain_st3_w_ovis_IS_s3/retrain_st3_w_ovis_IS_s3_mask2former_105000.pth")
# parser.add_argument('--IS_model', help='Path to pretrained network weight only',
#                     default=None)
parser.add_argument('--expand_mask', action='store_true')
parser.add_argument('--use_pixel_decoder', action='store_true')
# parser.add_argument('--share_backbone', default=True)
parser.add_argument('--share_backbone', action='store_true')

parser.add_argument('--fusion_mode', action='store_true')
# parser.add_argument('--fusion_model_path', default='/dfs/data/VOS/XMem/saves/XMem-s012.pth')
parser.add_argument('--train', default=False)

# Data options

parser.add_argument('--vot_path', default='/dfs/data/VOS/vots2023')
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument('--generic_path')

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
        
# Long-term memory options
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

# Multi-scale options
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=-1, type=int,
            help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']
config['use_IS'] = args.IS_model is not None


"""
Data preparation
"""

torch.autograd.set_grad_enabled(False)

# Load our checkpoint
network = XMem(config, args.model).cuda().eval()
mask2former_model = None

if args.model is not None:
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)

    #  IS model
    if args.IS_model is not None:
        mask2former_model = MaskFormerModel(config).cuda().eval()
        model_weights = torch.load(args.IS_model)
        mask2former_model.load_parallal_model(model_weights)
else:
    print('No model loaded.')

# if config['fusion_mode']:
#     assert config['fusion_model_path'] is not None
#     network_fusion = XMem(config, config['fusion_model_path']).cuda().eval()
#     model_weights = torch.load(config['fusion_model_path'])
#     network_fusion.load_weights(model_weights, init_as_zero_if_needed=True)


total_process_time = 0
total_frames = 0


# VOT-toolkit
handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()

imagefile = handle.frame()
video = imagefile.split("/")[-3]
# image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
# # print(image.shape)
# # print(image.max())
# trackers = [NCCTracker(image, object) for object in objects]

# while True:
#     imagefile = handle.frame()
#     if not imagefile:
#         break
#     image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
#     handle.report([tracker.track(image) for tracker in trackers])

# Start eval
data_root = args.vot_path
with open(path.join(data_root, 'sequences', 'list.txt')) as f:
    vid_list = sorted([line.strip() for line in f])

image_dir = path.join(data_root, 'sequences')
mask_dir = path.join(data_root, 'Annotations')
vid_reader = VideoReader(video,
                path.join(image_dir, video, 'color'),
                path.join(mask_dir, video),
                size=config['size'],
                size_dir=path.join(image_dir, video, 'color'),
            )
loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
vid_name = vid_reader.vid_name
vid_length = len(loader)

# if vid_name == "giraffe-15":
    # import pdb
    # pdb.set_trace()
# no need to count usage for LT if the video is not that long anyway
# 基本 500 帧左右要用到 LT
config['enable_long_term_count_usage'] = (
    config['enable_long_term'] and
    (vid_length
        / (config['max_mid_term_frames']-config['min_mid_term_frames'])
        * config['num_prototypes'])
    >= config['max_long_term_elements']
)

mapper = MaskMapper()
processor = InferenceCore(network, mask2former_model=mask2former_model, config=config)
# if config['fusion_mode']:
#     processor_fusion = InferenceCore(network_fusion, mask2former_model=None, config=config)
first_mask_loaded = False

def Prob_index_mask(out_mask, objects):


    result = []
    labels = np.unique(out_mask)
    labels = labels[labels != 0]

    for i in range(len(objects)):
        new_mask = np.zeros((out_mask.shape[-2:]), dtype=np.uint8)
        this_mask = (out_mask == i + 1)
        new_mask[this_mask] = 1
        result.append(new_mask)
    return result

for ti, data in enumerate(loader):
    with torch.cuda.amp.autocast(enabled=not args.benchmark):
        rgb = data['rgb'].cuda()[0]
        msk = data.get('mask')
        info = data['info']
        frame = info['frame'][0]
        shape = info['shape']
        need_resize = info['need_resize'][0]

        # VOT
        if ti != 0:
            imagefile = handle.frame()
        assert imagefile.split("/")[-1] == frame

        new_objects = []
        for ob in objects:
            new_ob_mask = np.zeros(rgb.shape[-2:], dtype=np.uint8)
            new_ob_mask[:ob.shape[0], :ob.shape[1]] = ob
            new_objects.append(new_ob_mask)

        new_mask = np.zeros((rgb.shape[-2], rgb.shape[-1]), dtype=np.uint8)
        for i, nm in enumerate(new_objects):
            this_mask = (nm == 1)
            new_mask[this_mask] = i + 1
        # print(new_mask.shape)
        # print(new_mask.max())
        """
        For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        Seems to be very similar in testing as my previous timing method 
        with two cuda sync + time.time() in STCN though 
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if not first_mask_loaded:
            if msk is not None:
                first_mask_loaded = True
            else:
                # no point to do anything without a mask
                continue

        if args.flip:
            rgb = torch.flip(rgb, dims=[-1])
            msk = torch.flip(msk, dims=[-1]) if msk is not None else None

        # Map possibly non-continuous labels to continuous ones
        if msk is not None:
            # convert ori msk to one-hot label (1, H, W) -> (num_object, H, W)

            # VOT
            msk, labels = mapper.convert_mask(new_mask)
            # print(msk.shape)
            # print(msk.max())
            msk = torch.Tensor(msk).cuda()
            if need_resize:
                msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
            processor.set_all_labels(list(mapper.remappings.values()))
            # if config['fusion_mode']:
            #     processor_fusion.set_all_labels(list(mapper.remappings.values()))
        else:
            labels = None

        # Run the model on this frame
        # shape:[num_objects + bg(1), H, W]
        if config['fusion_mode']:
            # prob_fusion, logits_fusion = processor_fusion.step(rgb, msk, labels, end=(ti == vid_length - 1))
            prob, _ = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))
            prob = (prob.detach().cpu().numpy() * 255).astype(np.uint8)

            f = open(os.path.join("/dfs/data/VOS/XMem/output/vots2023/fusion_test/Scores", vid_name, frame[:-4] + '.pkl'), 'rb')
            prob_fusion = pkl.load(f)

            prob = 0 * prob + 0.5 * prob_fusion

        else:

            prob, _ = processor.step(rgb, msk, labels, end=(ti == vid_length - 1))
        # Upsample to original size if needed
        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

        end.record()
        torch.cuda.synchronize()
        total_process_time += (start.elapsed_time(end)/1000)
        total_frames += 1

        if args.flip:
            prob = torch.flip(prob, dims=[-1])

        # Probability mask -> index mask
        if config['fusion_mode']:
            out_mask = np.argmax(prob, axis=0).astype(np.uint8)
        else:
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        result = Prob_index_mask(out_mask, objects)
        # print(len(result))

        handle.report(result)

print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
