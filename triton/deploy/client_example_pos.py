import os
import glob
import argparse

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./triton'))
import utils.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Inference YoloV8-Pose")
    parser.add_argument("--img_dir", default='./ultralytics/ultralytics/assets', type=str)
    parser.add_argument("--model_name", default='yolov8_pose_ensemble')
    return parser.parse_args()


class YoloV8PoseAPI:
    def __init__(self, host: str = 'localhost', port: int = 8001, model_name: str = 'yolov8_kpts_ensemble') -> None:
        self.host = host
        self.port = port
        self.model_name = f'{model_name}'
        self.url = f'{self.host}:{self.port}'
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.url,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
        config = self.triton_client.get_model_config(model_name, as_json=True)
        if len(config['config']['input'][0]['dims']) >= 3:
            self.channels = int(config['config']['input'][0]['dims'][0])
            self.height = int(config['config']['input'][0]['dims'][-2])
            self.width = int(config['config']['input'][0]['dims'][-1])
            self.input_name = config['config']['input'][0]['name']
        else:
            raise NotImplementedError()

        self.outputs = []
        for output in config['config']['output']:
            self.outputs.append(grpcclient.InferRequestedOutput(output['name']))
        self.model_dtype = "FP32"
        print(config)

    def _preproc(self, image_batch):
        # preproc №2
        return utils.preproc_img_for_inference(image_batch)

    def predict(self, image_batch):
        batch = self._preproc(np.array(image_batch))
        inputs = [grpcclient.InferInput(self.input_name, list(batch.shape), self.model_dtype)]
        inputs[0].set_data_from_numpy(batch)
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.outputs,
            headers={},
            compression_algorithm=None)
        return (results.as_numpy('result_bboxes').copy(), results.as_numpy('result_scores').copy(), results.as_numpy('result_landmarks').copy())


def draw_detections(img, boxes, scores, lands):
    for box_idx in range(len(boxes)):
        bbox = boxes[box_idx]
        land = lands[box_idx]
        score = scores[box_idx]

        print(f'\tbox idx: {box_idx},\tscore: {score},\tbox: {bbox},\tland: {land}')
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


def decode_detection(detections):
    boxes = detections[0]
    scores = detections[1]
    landms = detections[2]

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


def main(args) -> None:
    model_api = YoloV8PoseAPI(model_name=args.model_name)
    img_h, img_w = model_api.height, model_api.width
    image_list = utils.load_images(img_dir=args.img_dir)
    
    # preproc №1 !!!
    image_list = [utils.resize_with_padding(img, new_shape=(img_h, img_w)) for img in image_list]

    detections = model_api.predict(image_list)
    batch_boxes, batch_scores, batch_landms = decode_detection(detections)

    num_batches = len(batch_boxes)

    for idx_batch in range(num_batches):
        print('\nidx_batch:', idx_batch)
        boxes = batch_boxes[idx_batch]
        scores = batch_scores[idx_batch]
        landms = batch_landms[idx_batch]
        img = draw_detections(
            image_list[idx_batch],
            boxes,
            scores,
            landms,
        )
        utils.imshow(img)


if __name__ == "__main__":
    args = parse_args()
    main(args)
