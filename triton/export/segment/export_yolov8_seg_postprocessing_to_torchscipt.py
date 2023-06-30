import argparse
import os
import cv2

import torch
import torch.nn.functional as F
import torchvision

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from random import shuffle, randint
from typing import Any, Tuple, List


import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./triton'))
import utils.utils as utils


# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/v8/segment/predict.py#L16
class PostprocessingYoloV8Mask(torch.nn.Module):
    def __init__(self, conf: float, iou: float, max_det: int, img_h: int, img_w: int, nc: int, output_mask_h: int, output_mask_w: int, max_nms: int, max_wh: int, filtred_classes: List[int]) -> None:
        super(PostprocessingYoloV8Mask, self).__init__()
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.img_h = img_h
        self.img_w = img_w
        self.shape = (img_h, img_w)
        self.max_nms = max_nms
        self.max_wh = max_wh
        self.nc = nc
        self.output_mask_h = output_mask_h
        self.output_mask_w = output_mask_w
        self.filtred_classes = filtred_classes  # 0-person, 5-bus, .etc

    def _clip_boxes(self, boxes):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
          boxes (torch.Tensor): the bounding boxes to clip
          shape (tuple): the shape of the image
        """
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, self.shape[1])  # x1
            boxes[..., 1].clamp_(0, self.shape[0])  # y1
            boxes[..., 2].clamp_(0, self.shape[1])  # x2
            boxes[..., 3].clamp_(0, self.shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(
                0, self.shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(
                0, self.shape[0])  # y1, y2

    def _box_iou(self, box1, box2, eps: float = 1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(
            2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def _scale_boxes(self, boxes, ratio_pad=None):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
          boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
          ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                             calculated based on the size difference between the two images.

        Returns:
          boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(self.shape[0] / self.shape[0],
                       self.shape[1] / self.shape[1])
            pad = (self.shape[1] - self.shape[1] * gain) / \
                2, (self.shape[0] - self.shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self._clip_boxes(boxes)
        return boxes

    def _crop_mask(self, masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

        Args:
          masks (torch.Tensor): [h, w, n] tensor of masks
          boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

        Returns:
          (torch.Tensor): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(
            boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
            None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
            None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def _process_mask(self, protos, masks_in, bboxes):
        """
        Apply masks to bounding boxes using the output of the mask head.

        Args:
            protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
            masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
            bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
            shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
            upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

        Returns:
            (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
                are the height and width of the input image. The mask is applied to the bounding boxes.
        """
        c, mh, mw = protos.shape  # CHW
        ih = self.img_h
        iw = self.img_w
        masks = (masks_in @ protos.float().view(c, -1)
                 ).sigmoid().view(-1, mh, mw)
        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = torch.unsqueeze(self._crop_mask(
            masks, downsampled_bboxes), 0)  # CHW
        if self.output_mask_w and self.output_mask_h and self.output_mask_w != mw and self.output_mask_h != mh:
            masks = F.interpolate(masks, (self.output_mask_h, self.output_mask_w),
                                  mode='bilinear', align_corners=False)  # CHW
        return masks.gt_(0.5).to(torch.uint8)

    def _xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py#L136
    def non_max_suppression(self, prediction, proto):
        device = prediction.device
        bs = prediction.shape[0]  # batch size
        nm = prediction.shape[1] - self.nc - 4
        mi = 4 + self.nc

        xc = prediction[:, 4:mi].amax(1) > self.conf  # candidates

        boxes_output = torch.zeros(
            prediction.shape[0], self.max_det, 4, device=device)
        scores_output = torch.zeros(
            prediction.shape[0], self.max_det, 1, device=device)
        label_output = torch.zeros(
            prediction.shape[0], self.max_det, 1, device=device)

        if self.output_mask_h and self.output_mask_w and self.output_mask_h != proto.shape[2] and self.output_mask_w != proto.shape[3]:
            masks_output_global = torch.zeros(
                bs, self.max_det, self.output_mask_h, self.output_mask_w, device=device)
        else:
            masks_output_global = torch.zeros(
                bs, self.max_det, proto.shape[2], proto.shape[3], device=device)

        filtred_classes = torch.tensor(self.filtred_classes, device=device)

        for xi, x in enumerate(prediction):  # image index, image inference
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            box, cls, mask = x.split((4, self.nc, nm), 1)
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = self._xywh2xyxy(box)

            scores, label = cls.max(1, keepdim=True)

            if filtred_classes[0] >= 0:  # -1 for all classes
                indexes = (label == filtred_classes).any(1)
                box = box[indexes]
                scores = scores[indexes]
                label = label[indexes].float()
                mask = mask[indexes]

            # Batched NMS
            c = label * self.max_wh  # classes
            boxes = box + c  # boxes (offset by class)
            indexes = torchvision.ops.nms(
                boxes, scores.view(-1), self.iou)  # NMS
            top_k = min(box[indexes].shape[0], self.max_det)

            boxes_output[xi, :top_k] = box[indexes]
            scores_output[xi, :top_k] = scores[indexes].view(-1, 1)
            label_output[xi, :top_k] = label[indexes].view(-1, 1)
            masks_output_global[xi, :top_k] = self._process_mask(
                proto[xi], mask[indexes], box[indexes])[0, :]
        return boxes_output, scores_output, label_output, masks_output_global

    def forward(self, preds: torch.Tensor, proto: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        preds_nms = self.non_max_suppression(preds, proto)
        return preds_nms[0],  preds_nms[1], preds_nms[2], preds_nms[3]


class OnnxInferenceYoloSeg:
    def __init__(self, path_onnx_file) -> None:
        self.onnx_inference_sess = ort.InferenceSession(
            path_onnx_file, providers=['CPUExecutionProvider'])

    def _preproc(self, image_batch):
        # preproc №2
        return utils.preproc_img_for_inference(image_batch)

    def _inference(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.onnx_inference_sess.run(None, {'images': x})

    def __call__(self, images: list) -> Any:
        x = self._preproc(images)
        x = self._inference(x)
        return x


def draw_detections(img, boxes, scores, labels, masks):
    for box_idx in range(len(boxes)):
        bbox = boxes[box_idx]
        label = int(labels[box_idx].item())

        score = scores[box_idx]
        mask = masks[box_idx] * 255
        mask = cv2.resize(mask, (640, 640))
        mask = np.clip(mask, 0, 255).astype(np.uint8)

        print(
            f'\tbox idx: {box_idx},\tscore: {score},\tbox: {bbox},\tlabel: {label},\tmask {mask.shape}')
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, f'{label}', (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        color = np.array((randint(0, 255), randint(
            0, 255), randint(0, 255)), dtype='uint8')
        masked_img = np.where(mask[..., None], color, img)
        print(img.shape, mask.shape, masked_img.shape)
        img = cv2.addWeighted(img, 0.3, masked_img, 0.7, 0)
    return img


def decode_detection(detections):
    boxes = detections[0].cpu().numpy()
    scores = detections[1].cpu().numpy()
    labels = detections[2].cpu().numpy()
    masks = detections[3].cpu().numpy()

    batch_boxes = []
    batch_scores = []
    batch_labels = []
    batch_masks = []

    for idx_batch in range(len(boxes)):
        # skip nondetections (scores == 0)
        non_zero = np.where(scores[idx_batch] > 0)[0]
        batch_boxes.append(boxes[idx_batch][non_zero])
        batch_scores.append(scores[idx_batch][non_zero])
        batch_labels.append(labels[idx_batch][non_zero])
        batch_masks.append(masks[idx_batch][non_zero])

    return (batch_boxes, batch_scores, batch_labels, batch_masks)


def main(args):
    yolo_seg_model = OnnxInferenceYoloSeg(args.path_onnx_file)
    
    image_list = utils.load_images(img_dir=args.img_dir)

    # preproc №1 !!!
    image_list = [utils.resize_with_padding(img, new_shape=(args.img_h, args.img_w)) for img in image_list]

    yolo_output_preds, yolo_output_proto = yolo_seg_model(image_list)

    postproc_model = PostprocessingYoloV8Mask(
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        img_h=args.img_h,
        img_w=args.img_w,
        output_mask_h=args.output_mask_h,
        output_mask_w=args.output_mask_w,
        nc=args.nc,
        max_nms=args.max_nms,
        max_wh=args.max_wh,
        filtred_classes=args.filtred_classes,
    )
    yolo_output_preds = torch.from_numpy(yolo_output_preds).cuda()
    yolo_output_proto = torch.from_numpy(yolo_output_proto).cuda()

    postproc_model.eval().cuda().half()
    with torch.jit.optimized_execution(True):
        traced_postprocessing = torch.jit.script(postproc_model)

    traced_postprocessing.cuda()
    detections = traced_postprocessing(yolo_output_preds, yolo_output_proto)
    batch_boxes, batch_scores, batch_labels, batch_masks = decode_detection(
        detections)
    traced_postprocessing.save(args.path_output_model)

    num_batches = len(batch_boxes)

    for idx_batch in range(num_batches):
        print('\nidx_batch:', idx_batch)
        boxes = batch_boxes[idx_batch]
        scores = batch_scores[idx_batch]
        labels = batch_labels[idx_batch]
        masks = batch_masks[idx_batch]

        print('masks', masks.shape)

        img = draw_detections(
            image_list[idx_batch],
            boxes,
            scores,
            labels,
            masks
        )
        utils.imshow(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Export postprocessing for instance segmentation to torchscipt')
    parser.add_argument('--conf', default=0.2, type=float,
                        help='score threshold for nms')
    parser.add_argument('--iou', default=0.5, type=float,
                        help='iou threshold for nms')
    parser.add_argument('--max_det', default=100, type=int,
                        help='count output detection per batch')
    parser.add_argument('--img_w', default=640, type=int,
                        help='input image w used in export torch model to onnx')
    parser.add_argument('--img_h',  default=640, type=int,
                        help='input image h used in export torch model to onnx')
    parser.add_argument('--max_nms', default=30000, type=int)
    parser.add_argument('--max_wh', default=7680, type=int)
    # coco classes, 0-person, 5-bus, -1 for all objects,  --filtred_classes=0,5,27
    parser.add_argument('--filtred_classes', default='0',
                        type=str, help='ids for keeping after postporcessing')
    # num classes, 80 - for coco
    parser.add_argument('--nc', default=80, type=int,
                        help='num classes, used from train')
    # for resize h output mask, 0 - not resize to img_h
    parser.add_argument('--output_mask_h', default=0, type=int,
                        help='output mask h, if h > 0, it will be resize to h, default output_mask_h < img_h')
    # for resize w output mask, 0 - not resize to img_w
    parser.add_argument('--output_mask_w', default=0, type=int,
                        help='output mask w, if w > 0, it will be resize to w, default output_mask_w < img_w')
    parser.add_argument('--path_output_model',
                        default='triton/deploy/model_repository/yolov8_seg_postprocessing/1/model.pt', type=str)
    parser.add_argument(
        '--img_dir', default='./ultralytics/ultralytics/assets', type=str)
    # $ yolo export model=yolov8m-seg.pt format=onnx dynamic=True
    parser.add_argument('--path_onnx_file',
                        default='./yolov8m-seg.onnx', type=str)

    args = parser.parse_args()
    try:
        args.filtred_classes = [int(cl)
                                for cl in args.filtred_classes.split(',')]
    except:
        args.filtred_classes = [-1]

    main(args)
