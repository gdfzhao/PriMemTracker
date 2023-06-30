import os
import json
from pycocotools.coco import COCO

subsets = ['train']
for subset in subsets:
    odgt_file = {}
    instance_json_path = '/dfs/data/VOS/COCO/annotations/' + 'instances_{}2017.json'.format(subset)
    coco = COCO(instance_json_path)
    imgs = coco.loadImgs(coco.getImgIds())
    info = {}
    for img in imgs:
        info[img['file_name']] = {'height': img['height'], 'width': img['width']}

    with open("{}_coco.odgt".format(subset), 'a') as f:
        len_split = len(os.listdir("/dfs/data/VOS/COCO/Annotations/{}".format(subset)))
        result = []
        for imgfile in os.listdir("/dfs/data/VOS/COCO/Annotations/{}".format(subset)):
            if not imgfile.endswith("png"): continue
            jpg_name = imgfile.replace('png', 'jpg')
            dict = {"fpath_img": "COCO/{}2017/{}".format(subset, jpg_name), "fpath_segm": "COCO/Annotations/{}/{}".format(subset, imgfile),
                   "width": info[jpg_name]['width'], "height": info[jpg_name]['height']}
            str = json.dumps(dict)
            result.append(dict)
            f.write(str + '\n')
    f.close()
