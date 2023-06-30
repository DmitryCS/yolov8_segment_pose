import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import glob
from pathlib import Path

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/dataloaders/v5augmentations.py#L116
def resize_with_padding(img, new_shape):
    # оригинальный препроцесс, без него падает точность модели
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # print(img.shape)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # print('top, bottom, left, right', top, bottom, left, right)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    return img #, [top, bottom, left, right]

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/data/dataloaders/v5augmentations.py#L116
def resize_with_padding_and_borders(img, new_shape):
    # оригинальный препроцесс, без него падает точность модели
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # print(img.shape)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # print('top, bottom, left, right', top, bottom, left, right)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    return img, [top, bottom, left, right]

def preproc_img_for_inference(x):
    x = np.array(x).astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)  # hwc -> bhwc
    x = x.transpose((0, 3, 1, 2))  # bhwc to bchw,
    return x/255.0  # [0,255] -> [0,1]


def read_img_bgr(img_path):
    img = cv2.imread(img_path)  # bgr
    return img


def load_images(img_dir) -> list:
    image_paths_list = os.listdir(img_dir)
    # shuffle for check models on different sequence images!!!
    shuffle(image_paths_list)
    images = []
    for img_path in image_paths_list:
        img_path = os.path.join(img_dir, img_path)
        images.append(read_img_bgr(img_path))
    return images

def yolo_to_coco_box(box):
    x_center, y_center, width_box, height_box = box
    x_left_up, y_left_up = x_center - width_box / 2, y_center - height_box / 2
    x_right_bottom, y_right_bottom = x_left_up + width_box, y_left_up + height_box
    return [x_left_up, y_left_up, x_right_bottom, y_right_bottom]

def load_dataset(dataset_dir):
    dataset_part = 'val2017'
    path_labels = f'{dataset_dir}/labels/{dataset_part}'    
    path_images = f'{dataset_dir}/images/{dataset_part}'    
    labels_paths = glob.glob(f'{path_labels}/*.txt') 
    images, labels_gt = [], []
    for labels_path in labels_paths[:3]:
        image_name = labels_path.split('/')[-1].split('.')[0]                
        with open(labels_path, 'r') as f:
            segpose_data = f.readlines()        
        boxes = []
        keypoints = []
        segmentation = []
        for line in segpose_data:
            line_split = line.split()
            line_split_float = list(map(float, line_split))                      
            box_yolo = line_split_float[1:5]    
            box_coco = yolo_to_coco_box(box_yolo)            
            boxes.append(box_coco)           
            keypoints.append(line_split_float[5:5+3*17])
            segmentation.append(line_split_float[5+3*17:])
        
        labels_gt.append([boxes, segmentation, keypoints])
        img_path = f'{path_images}/{image_name}.jpg'
        images.append(read_img_bgr(img_path))        

    return images, labels_gt


def imshow(img, title='img'):
    plt.imshow(img)
    plt.title(title)
    plt.show()

def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor) or (numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
    """
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False):
    """
    Rescale segment coordinates (xyxy) from img1_shape to img0_shape

    Args:
      img1_shape (tuple): The shape of the image that the coords are from.
      coords (torch.Tensor): the coords to be scaled
      img0_shape (tuple): the shape of the image that the segmentation is being applied to
      ratio_pad (tuple): the ratio of the image size to the padded image size.
      normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False

    Returns:
      coords (torch.Tensor): the segmented image.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., 0] -= pad[0]  # x padding
    coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords
