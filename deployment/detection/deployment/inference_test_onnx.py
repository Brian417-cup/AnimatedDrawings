# Run following command for onnx export:
# python inference_test_onnx.py --model_path <your_*.onnx_path> --img_path <img_path>
import os

import numpy as np
import onnxruntime as ort
import time
import cv2
import argparse


def parser_args():
    parser = argparse.ArgumentParser(description="onnx_inference")
    parser.add_argument('--img_path', help="The picture that you want to inference")
    parser.add_argument('--model_path', required=True, help="The onnx model path that you want to use")
    args = parser.parse_args()
    return args


def preprocess(img):
    norm_mean = [103.53, 116.28, 123.675]
    norm_std = [1.0, 1.0, 1.0]
    to_rgb = False
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad to mod by 32 for both width and height
    h, w = data.shape[:2]
    h, w = [int((_ * 1.5) // 32 * 32) for _ in [h, w]]
    h, w = min(1344, h), min(1344, w)
    # normalize
    data = (data - norm_mean) / norm_std
    # [H,W,C] -> [C,H,W]
    data = np.transpose(data, (2, 0, 1)).astype(np.float32)

    # For mmdet MaskRCNN, this operation doesn't support
    # data = data.astype(np.float32) / 255.0
    data = np.expand_dims(data, axis=0)
    return data


###########################################################################################
# numpy nms
def numpy_nms(boxes: np.array, scores: np.array, iou_threshold: float):
    '''

    Args:
        boxes: [num_det,x1,y1,x2,y2]
        scores: [num_det]
        iou_threshold: thrsh_score

    Returns:
        All possible indicies after NMS filtered.

    Example:
        box =  np.array([[2,3.1,7,5],[3,4,8,4.8],[4,4,5.6,7],[0.1,0,8,1]])
        score = np.array([0.5, 0.3, 0.2, 0.4])

        indicies = numpy_nms(boxes=box, scores=score, iou_threshold=0.3)
    '''

    def box_area(boxes: np.array):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(box1: np.array, box2: np.array):
        area1 = box_area(box1)  # N
        area2 = box_area(box2)  # M
        # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep


###########################################################################################

def single_thresh_filter(bboxes_list: list, labels_list: list, masks_list: list, score_thr=0.3):
    '''

    Args:
        bboxes_list: [num_det,x1,y1,x2,y2,score]
        labels_list: [num_det]
        masks_list: [num_dete,h,w]

    Returns:
        filtered with certain threshold score where are score > score threshold

        bboxes_list: [filtered_num_det,x1,y1,x2,y2,score]
        labels_list: [filtered_num_det]
        masks_list: [filtered_num_dete,h,w]
    '''

    needindexs = np.where(bboxes_list[:, -1] > score_thr)
    bboxes = bboxes_list[needindexs]
    labels = labels_list[needindexs]
    masks = masks_list[needindexs]

    return bboxes, labels, masks


def main():
    args = parser_args()
    onnx_file = args.model_path
    img = cv2.imread(args.img_path)

    sess = ort.InferenceSession(onnx_file)
    start = time.time()

    input_data = preprocess(img)
    onnx_results = sess.run(None, {'input': input_data})
    end = time.time()
    print("Inference time:", end - start, "s")
    # for single inference, here we choose 1st for the single batch
    bboxes_list = onnx_results[0][0]
    labels_list = onnx_results[1][0]
    masks_list = onnx_results[2][0]

    # filter output with NMS threshold
    bboxes_list, labels_list, masks_list = single_thresh_filter(bboxes_list, labels_list, masks_list,
                                                                score_thr=0.2)

    scores_list = bboxes_list[:, -1].reshape(-1)
    bboxes_list = bboxes_list[:, :-1]

    nms_indicies = numpy_nms(bboxes_list, scores_list, iou_threshold=0.3)

    bboxes, masks, labels = bboxes_list[nms_indicies], masks_list[nms_indicies], labels_list[nms_indicies]

    for index, (box, mask) in enumerate(zip(bboxes, masks)):
        x1, y1, x2, y2 = box.astype(np.int32)
        mask = mask * 255
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        black = np.zeros_like(img)
        black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        black = mask.astype(np.uint8)

        _, contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 1:
            contours = np.array(contours).reshape((-1, 1, 2))
            cv2.polylines(img, [contours], isClosed=True, color=(0, 0, 255), thickness=3)
        else:
            for contour in contours:
                contour = np.array(contour).reshape((-1, 1, 2))
                cv2.polylines(img, [contour], isClosed=True, color=(0, 0, 255), thickness=3)

        alpha = 0.8

        mask = mask.astype(bool)
        random_colors = np.array([0, 255, 255])
        img[mask] = img[mask] * (1 - alpha) + random_colors * alpha

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
