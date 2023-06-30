import glob
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from parallel_work import parallel_work
import re
from moviepy.editor import ImageSequenceClip as imageSeq
from argparse import ArgumentParser

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--save_with_ori_img', default=True)
parser.add_argument('--save_video', action='store_true')
parser.add_argument('--dataset', help='vots2023/vot-test', default='vots2023')
parser.add_argument('--exp_dir', default=None, type=str)
parser.add_argument('--contrast_dir', help='Inputs are the exp dirs, split with , ', type=str, default=None)

args = parser.parse_args()
print(args)
SAVE_WITH_ORI_IMG = args.save_with_ori_img
SAVE_VIDEO = args.save_video
DATASET = args.dataset
EXP_DIR = args.exp_dir
CONTRAST_DIR = args.contrast_dir
CONTRAST_MODE = False
if CONTRAST_DIR is not None:
    CONTRAST_MODE = True
    CONTRAST_DIR = CONTRAST_DIR.split(',')
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

def visualization_pred_files(label_files, target_dir):
    img_array = []

    for root, file in label_files:
        if root.endswith('checkpoints'):
            continue
        video_name = root.split('/')[-1]
        os.makedirs(os.path.join(target_dir, video_name), exist_ok=True)
        raw_mask = cv2.imread(os.path.join(root, file), 0)
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
        if SAVE_WITH_ORI_IMG:
            ori_img = cv2.imread(os.path.join('../' + DATASET, 'sequences', video_name, 'color', file.replace('png', 'jpg')))
            color_mask = ori_img + color_mask * 0.5
        cv2.imwrite(os.path.join(target_dir, video_name, file), color_mask)

def visualization(vot_root):
    target_dir = os.path.join("output", DATASET, EXP_DIR + '_vis')
    pred_files = []
    for root, dirs, files in os.walk(vot_root):
        for file in tqdm(files, total=len(files)):
            if file.endswith('.png'):
                if os.path.join(root, file).split('/')[-2] in video_names:
                        pred_files.append((root, file))

    # visualization_pred_files(pred_files, target_dir)
    parallel_work(pred_files, visualization_pred_files, target_dir)

def img2video(vis_root, save_dir):
    video_files = []
    os.makedirs(save_dir, exist_ok=True)
    for root, dirs, files in os.walk(vis_root):
        # for video_name in tqdm(dirs):
        for video_name in video_names:

            if video_name.endswith('checkpoints'):
                continue
            video_files.append((root, video_name))
    parallel_work(video_files, img2video_files, save_dir)
    # img2video_files(video_files, save_dir)

def img2video_files(video_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for root, video_name in video_files:
            if os.path.exists(os.path.join(save_dir, '{}.mp4'.format(video_name))):
                continue
            with open(os.path.join('../', DATASET, 'sequences', video_name, 'sequence'), 'r') as f:
                txt = f.readlines()
                for line in txt:
                    if re.match("fps", line) is not None:
                        fps = int(line[4:])
                        break
            if not fps:
                raise NotImplementedError
            print("Start Generating Video:{} with FPS {}".format(video_name, fps))
            # 其它格式的图片也可以
            imgs_path = sorted(glob.glob(os.path.join(root, video_name, "*.png")))
            if len(imgs_path) > 0:
                imageSeq(sorted(glob.glob(os.path.join(root, video_name, "*.png"))), fps=fps).write_videofile(
                    os.path.join(save_dir, '{}.mp4'.format(video_name)))
            # img_array = []
            # for filename in glob.glob(os.path.join(root, video_name, '*.png')):
            #     img = cv2.imread(filename)
            #     height, width, layers = img.shape
            #     size = (width, height)
            #     img_array.append(img)

            # avi：视频类型，mp4也可以
            # cv2.VideoWriter_fourcc(*'DIVX')：编码格式
            # 5：视频帧率
            # size:视频中图片大小
            # import pdb
            # pdb.set_trace()
            # out = cv2.VideoWriter(os.path.join(save_dir, '{}.avi'.format(video_name)),
            #                       -1,
            #                       fps, size)
            #
            # for i in range(len(img_array)):
            #     out.write(img_array[i])
            # out.release()
def contrast_videos_files(video_names, CONTRAST_DIR, target_dir):
    for video_name in video_names:
        one_shot_cv_paths = [os.path.join("output", DATASET, exp_dir + "_videos", video_name + ".mp4")
                             for exp_dir in CONTRAST_DIR]
        # p1, p2, p3 = [os.path.join('output', 'vot-test', 'ori_xmem_pred_videos', video_name + '_mask.mp4'),
        #               os.path.join('output', 'vot-test', 'finetune_xmem_pred_videos', video_name + '.mp4'),
        #               os.path.join('output', 'vot-test', 'retrain_xmex_pred_videos', video_name + '.mp4')]
        clips = [VideoFileClip(p) for p in one_shot_cv_paths]
        cv_video = clips_array([clips])
        cv_video = cv_video.speedx(0.75)

        position = ['left', 'right']
        txt_clip = []
        if len(CONTRAST_DIR) > 2:
            raise NotImplementedError
        for i in range(len(CONTRAST_DIR)):
            txt_clip.append(TextClip(CONTRAST_DIR[i], fontsize=80, color='white',
                                     font='STHeitiMedium.ttc').set_pos((position[i], 'top')).set_duration(cv_video.duration))
        cv_video = CompositeVideoClip([cv_video] + txt_clip)

        cv_video.write_videofile(os.path.join(target_dir, video_name + '.mp4'))

with open(os.path.join('../', DATASET, 'sequences', "list_test.txt"), 'r') as f:

    vlist = f.readlines()
    video_names = []
    for v in vlist:
        imgpath = os.path.join(v.strip())
        video_names.append(imgpath)


# visualization(os.path.join("output", DATASET, EXP_DIR))
if SAVE_VIDEO:
    save_dir = os.path.join("output", DATASET, EXP_DIR + '_videos')
    img2video(os.path.join("output", DATASET, EXP_DIR), save_dir)
if CONTRAST_MODE:
    from moviepy.editor import *
    from moviepy.video.VideoClip import TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.io.VideoFileClip import VideoFileClip

    target_dir = os.path.join("output", DATASET, "Contrast_results", "{}_via_{}".format(CONTRAST_DIR[0], CONTRAST_DIR[1]))
    os.makedirs(target_dir, exist_ok=True)

    contrast_video_length = len(CONTRAST_DIR)
    # parallel_work(video_names, contrast_videos_files, CONTRAST_DIR, target_dir)
    contrast_videos_files(video_names, CONTRAST_DIR, target_dir)


# save_dir = os.path.join("output", DATASET, EXP_DIR + '_videos')
# img2video(os.path.join("output", DATASET, EXP_DIR + '_vis'), save_dir)
# from moviepy.editor import *
# from moviepy.video.VideoClip import TextClip
# from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# from moviepy.video.io.VideoFileClip import VideoFileClip
#
# target_dir = os.path.join("output", DATASET, "Contrast_results")
# os.makedirs(target_dir, exist_ok=True)
#
# contrast_video_length = len(CONTRAST_DIR)
# # parallel_work(video_names, contrast_videos_files, CONTRAST_DIR, target_dir)
# contrast_videos_files(video_names, CONTRAST_DIR, target_dir)
