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

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset, VOTTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem, MaskFormerModel
from inference.inference_core import InferenceCore

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='./saves/XMem.pth')
parser.add_argument('--IS_model', help='Path to pretrained network weight only')
parser.add_argument('--expand_mask', action='store_true')
parser.add_argument('--train', default=False)

# Data options
parser.add_argument('--d16_path', default='../DAVIS/2016')
parser.add_argument('--d17_path', default='../DAVIS/2017')
parser.add_argument('--y18_path', default='../YouTube2018')
parser.add_argument('--y19_path', default='../YouTube')
parser.add_argument('--lv_path', default='../long_video_set')
parser.add_argument('--vot_path', default='../vot-test')
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument('--generic_path')

parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G/VOT-TEST', default='D17')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--output', default=None)
parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

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
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

if args.output is None:
    args.output = f'../output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')
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
"""
Data preparation
"""
is_youtube = args.dataset.startswith('Y')
is_davis = args.dataset.startswith('D')
is_lv = args.dataset.startswith('LV')
is_vot = args.dataset.startswith('VOT')
if is_youtube or args.save_scores:
    out_path = path.join(args.output, 'Annotations')
else:
    out_path = args.output

if is_youtube:
    if args.dataset == 'Y18':
        yv_path = args.y18_path
    elif args.dataset == 'Y19':
        yv_path = args.y19_path

    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif is_davis:
    if args.dataset == 'D16':
        if args.split == 'val':
            # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            meta_dataset = DAVISTestDataset(args.d16_path, imset='../../2017/trainval/ImageSets/2016/val.txt', size=args.size)
        else:
            raise NotImplementedError
        palette = None
    elif args.dataset == 'D17':
        if args.split == 'val':
            meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'trainval'), imset='2017/val.txt', size=args.size)
        elif args.split == 'test':
            meta_dataset = DAVISTestDataset(path.join(args.d17_path, 'test-dev'), imset='2017/test-dev.txt', size=args.size)
        else:
            raise NotImplementedError

elif is_vot:
    if args.dataset == 'VOT-TEST':
        # For vot dataset
        meta_dataset = VOTTestDataset(args.vot_path, size=args.size)
        palette = None
    elif args.dataset == 'VOTS2023':
        raise NotImplementedError

elif is_lv:
    if args.dataset == 'LV1':
        meta_dataset = LongTestDataset(path.join(args.lv_path, 'long_video'))
    elif args.dataset == 'LV3':
        meta_dataset = LongTestDataset(path.join(args.lv_path, 'long_video_x3'))
    else:
        raise NotImplementedError
elif args.dataset == 'G':
    meta_dataset = LongTestDataset(path.join(args.generic_path), size=args.size)
    if not args.save_all:
        args.save_all = True
        print('save_all is forced to be true in generic evaluation mode.')
else:
    raise NotImplementedError

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

config['use_IS'] = args.IS_model is not None
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

total_process_time = 0
total_frames = 0

def visualization(raw_mask, vid_name, frame, DATASET='vots2023'):


    h, w = raw_mask.shape[:2]
    # import pdb
    # pdb.set_trace()
    color_mask = np.zeros((h, w, 3))
    instance_ids = np.unique(raw_mask)[1:]
    # print(instance_ids)
    for instance_id in instance_ids:
        ci = int(instance_id) % len(PALETTE)
        color_mask[:, :, 0][raw_mask == instance_id] = PALETTE[ci][0]
        color_mask[:, :, 1][raw_mask == instance_id] = PALETTE[ci][1]
        color_mask[:, :, 2][raw_mask == instance_id] = PALETTE[ci][2]

    ori_img = cv2.imread(os.path.join('../' + DATASET, 'sequences', vid_name, 'color', frame.replace('png', 'jpg')))
    color_mask = ori_img + color_mask * 0.5
    return color_mask


# Start eval
for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=False):

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
    first_mask_loaded = False

    for ti, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

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
                msk, labels = mapper.convert_mask(msk[0].numpy())
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                labels = None

            # Run the model on this frame
            # import ipdb
            # ipdb.set_trace()
            # shape:[num_objects + bg(1), H, W]
            prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))

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
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if args.save_scores:
                prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            if args.save_all or info['save'][0]:
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                # visualization
                # import pdb
                # pdb.set_trace()
                out_mask = visualization(out_mask, vid_name, frame)
                cv2.imwrite(os.path.join(this_out_path, frame[:-4]+'.png'), out_mask)

                # out_img = Image.fromarray(out_mask)
                # if vid_reader.get_palette() is not None:
                #     out_img.putpalette(vid_reader.get_palette())
                # out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')


print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if not args.save_scores:
    if is_youtube:
        print('Making zip for YouTubeVOS...')
        shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
    elif is_davis and args.split == 'test':
        print('Making zip for DAVIS test-dev...')
        shutil.make_archive(args.output, 'zip', args.output)
