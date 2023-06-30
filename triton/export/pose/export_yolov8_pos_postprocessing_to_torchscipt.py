import argparse
import os
import cv2

import torch
import torch.nn.functional as F
import torchvision

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from random import shuffle
from typing import Any, Tuple, List

import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./triton'))
import utils.utils as utils

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/v8/pose/predict.py#L14


class PostprocessingYoloV8Pose(torch.nn.Module):
    def __init__(self, conf: float, iou: float, max_det: int, img_h: int, img_w: int, kpt_shape: tuple, max_nms: int) -> None:
        super(PostprocessingYoloV8Pose, self).__init__()
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.img_h = img_h
        self.img_w = img_w
        self.kpt_shape = kpt_shape
        self.shape = (img_h, img_w)
        self.max_nms = max_nms

    def forward(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes_output, scores_output, kpts_output = self.non_max_suppression(preds,
                                                                            self.conf,
                                                                            self.iou,
                                                                            max_det=self.max_det,
                                                                            max_nms=self.max_nms)
        return boxes_output, scores_output, kpts_output.view(preds.shape[0], self.max_det, *self.kpt_shape)

    def _clip_boxes(self, boxes):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
          boxes (torch.Tensor): the bounding boxes to clip
          shape (tuple): the shape of the image
        """
        boxes[..., 0].clamp_(0, self.shape[1])  # x1
        boxes[..., 1].clamp_(0, self.shape[0])  # y1
        boxes[..., 2].clamp_(0, self.shape[1])  # x2
        boxes[..., 3].clamp_(0, self.shape[0])  # y2

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

    def _clip_coords(self, coords):
        """
        Clip line coordinates to the image boundaries.

        Args:
            coords (torch.Tensor) or (numpy.ndarray): A list of line coordinates.
            shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

        Returns:
            (None): The function modifies the input `coordinates` in place, by clipping each coordinate to the image boundaries.
        """
        coords[..., 0].clamp_(0, self.shape[1])  # x
        coords[..., 1].clamp_(0, self.shape[0])  # y

    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py#L136
    def non_max_suppression(
            self,
            prediction,
            conf_thres: float,
            iou_thres: float,
            max_det: int,
            max_nms: int
    ):
        device = prediction.device
        bs = prediction.shape[0]  # batch size
        xc = prediction[:, 4] > conf_thres  # candidates

        boxes_output = torch.zeros(bs, max_det, 4, device=device)
        scores_output = torch.zeros(bs, max_det, 1, device=device)
        kpts_output = torch.zeros(bs, max_det, 51, device=device)

        for xi, x in enumerate(prediction):
            x = x.transpose(0, -1)[xc[xi]]

            if not x.shape[0]:
                continue

            boxes, scores, kpts = x[:, :4], x[:, 4], x[:, 5:]
            boxes = self._xywh2xyxy(boxes)
            self._clip_boxes(boxes)
            scores_indexes = scores.argsort(descending=True)[:max_nms]
            boxes = boxes[scores_indexes]
            scores = scores[scores_indexes]
            kpts = kpts[scores_indexes]
            self._clip_coords(kpts)

            nms_indexes = torchvision.ops.nms(boxes, scores, iou_thres)[
                :max_det]  # NMS
            top_k = min(boxes[nms_indexes].shape[0], max_det)

            boxes_output[xi, :top_k] = boxes[nms_indexes]
            scores_output[xi, :top_k] = scores[nms_indexes].view(-1, 1)
            kpts_output[xi, :top_k] = kpts[nms_indexes]

        return boxes_output, scores_output, kpts_output


class OnnxInferenceYoloPose:
    def __init__(self, path_onnx_file) -> None:
        self.onnx_inference_sess = ort.InferenceSession(
            path_onnx_file, providers=['CPUExecutionProvider'])

    def _inference(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.onnx_inference_sess.run(None, {'images': x})

    def _preproc(self, image_batch):
        # preproc №2
        return utils.preproc_img_for_inference(image_batch)

    def __call__(self, images: list) -> Any:
        x = self._preproc(images)
        x = self._inference(x)
        return x


def decode_detection(detections):
    boxes = detections[0].cpu().numpy()
    scores = detections[1].cpu().numpy()
    landms = detections[2].cpu().numpy()

    batch_boxes = []
    batch_scores = []
    batch_landms = []

    for idx_batch in range(len(boxes)):
        # skip nondetections (scores == 0)
        non_zero = np.where(scores[idx_batch] > 0)[0]
        batch_boxes.append(boxes[idx_batch][non_zero])
        batch_scores.append(scores[idx_batch][non_zero])
        batch_landms.append(landms[idx_batch][non_zero])

    return (batch_boxes, batch_scores, batch_landms)


def draw_detections(img, boxes, scores, lands):
    for box_idx in range(len(boxes)):
        bbox = boxes[box_idx]
        land = lands[box_idx]
        score = scores[box_idx]

        print(
            f'\tbox idx: {box_idx},\tscore: {score},\tbox: {bbox},\tland: {land}')
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (0, 0, 0), 2)
        for idx_kpt in range(len(land)):
            print(land[idx_kpt])
            x, y, v = land[idx_kpt]
            if v > 0.1:
                cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 4)
                cv2.putText(img, f'{idx_kpt}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 4)
                cv2.putText(img, f'{idx_kpt}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main(args):
    yolo_seg_model = OnnxInferenceYoloPose(args.path_onnx_file)
    image_list = utils.load_images(img_dir=args.img_dir)
    # preproc №1 !!!
    image_list = [utils.resize_with_padding(img, new_shape=(args.img_h, args.img_w)) for img in image_list]

    yolo_output_preds = yolo_seg_model(image_list)[0]

    postproc_model = PostprocessingYoloV8Pose(
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        img_h=args.img_h,
        img_w=args.img_w,
        kpt_shape=args.kpt_shape,
        max_nms=args.max_nms,
    )

    yolo_output_preds = torch.from_numpy(yolo_output_preds).cuda()

    postproc_model.eval().cuda().half()

    with torch.jit.optimized_execution(True):
        traced_postprocessing = torch.jit.script(postproc_model)
    traced_postprocessing.cuda()
    detections = traced_postprocessing(yolo_output_preds)
    batch_boxes, batch_scores, batch_kpts = decode_detection(detections)
    traced_postprocessing.save(args.path_output_model)

    num_batches = len(batch_boxes)

    for idx_batch in range(num_batches):
        print('\nidx_batch:', idx_batch)
        boxes = batch_boxes[idx_batch]
        scores = batch_scores[idx_batch]
        kpts = batch_kpts[idx_batch]

        print('kpts', kpts.shape)

        img = draw_detections(
            image_list[idx_batch],
            boxes,
            scores,
            kpts
        )
        utils.imshow(img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Export postprocessing for pose estimation to torchscipt')
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
    parser.add_argument('--kpt_shape', default=(17, 3),
                        type=tuple, help='kpts output shape, used from train')
    parser.add_argument('--max_nms', default=30000, type=int)
    parser.add_argument(
        '--img_dir', default='./ultralytics/ultralytics/assets', type=str)
    parser.add_argument('--path_output_model',
                        default='triton/deploy/model_repository/yolov8_pose_postprocessing/1/model.pt', type=str)
    # $ yolo export model=yolov8m-pose.pt format=onnx dynamic=True
    parser.add_argument('--path_onnx_file',
                        default='./yolov8m-pose.onnx', type=str)

    main(parser.parse_args())
