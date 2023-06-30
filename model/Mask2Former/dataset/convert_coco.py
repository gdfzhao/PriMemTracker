import cv2
import random
import json, os
import numpy as np
from pycocotools.coco import COCO

subset = 'train'
instance_json_path = '/dfs/data/VOS/COCO/annotations/' + 'instances_{}2017.json'.format(subset)
coco = COCO(instance_json_path)
catIds = coco.getCatIds()
imgIds = coco.getImgIds()
cats = coco.loadCats(catIds)
imgs = coco.loadImgs(imgIds)

none_list = []
img_list = os.listdir("/dfs/data/VOS/COCO/{}2017".format(subset))
for img in imgs:
    if img['file_name'] not in img_list: continue
    height, width = img['height'], img['width']
    frame_mask = np.zeros((height, width), dtype=np.uint8)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        if type(ann['segmentation']) != list or ann['segmentation'] is None:
            none_list.append(ann)
            print(img['file_name'])
            # continue
        m = coco.annToMask(ann)
        mask_id = ann['category_id']
        frame_mask = np.where(m > 0, mask_id, frame_mask)
    # import pdb
    #
    # pdb.set_trace()
    save_path = os.path.join("/dfs/data/VOS/COCO/Annotations/", subset, img['file_name'])
    cv2.imwrite(save_path.replace('jpg', 'png'), frame_mask)

print("Unknown segmentation:", len(none_list))
