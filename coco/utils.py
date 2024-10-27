import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def initialize_coco_api(ann_file):
    """用于加载COCO标注数据"""
    return COCO(ann_file)


def print_coco_info(coco):
    """用于打印COCO数据集中的分类和超分类等信息"""
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    print('COCO categories: {}\n'.format(' '.join(cat_nms)))

    sup_nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: {}'.format(' '.join(sup_nms)))

    # 获得分类的ID
    catIds_1 = coco.getCatIds(catNms=cat_nms, supNms=sup_nms)
    print(f'分类的ID : {catIds_1}')
    # 获得所有图片ID
    imgIds_1 = coco.getImgIds()
    print(f'所有图片ID : {imgIds_1}')
    # 打印图片数量
    print(f'图片的数量 ：{len(imgIds_1)}')


def draw_annotations(img, anns, coco):
    """绘制掩码、关键点和骨架"""
    for ann in anns:
        # Draw mask
        if 'segmentation' in ann:
            mask = coco.annToMask(ann).astype(np.bool_)  # Updated line here
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

        # Draw keypoints
        if 'keypoints' in ann and len(ann['keypoints']) % 3 == 0:
            keypoints = ann['keypoints']
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i:i+3]
                if v:
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

        # Draw skeleton
        if 'skeleton' in ann:
            skeleton = ann['skeleton']
            for limb in skeleton:
                start_idx, end_idx = limb[0]-1, limb[1]-1
                if keypoints[(start_idx*3)+2] and keypoints[(end_idx*3)+2]:
                    start_point = (keypoints[start_idx*3], keypoints[start_idx*3+1])
                    end_point = (keypoints[end_idx*3], keypoints[end_idx*3+1])
                    cv2.line(img, start_point, end_point, (0, 255, 0), 2)


def draw_rectangles(img, anns):
    """绘制bbox"""
    for j in range(len(anns)):
        coordinate = []
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        coordinate.append(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][1])
        # print(coordinate)
        left = np.rint(coordinate[0])
        right = np.rint(coordinate[1])
        top = np.rint(coordinate[2])
        bottom = np.rint(coordinate[3])
        # 左上角坐标, 右下角坐标
        cv2.rectangle(img,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      2)


def process_and_save_images(coco, img_ids, img_path, result_path):
    """用于处理给定的图片ID列表，并保存结果"""
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_name = img_info['file_name']
        img = cv2.imread(os.path.join(img_path, image_name))
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        draw_annotations(img, anns, coco)
        # draw_rectangles(img, anns)
        cv2.imwrite(os.path.join(result_path, image_name), img)
