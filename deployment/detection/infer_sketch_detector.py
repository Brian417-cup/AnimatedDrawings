'''
python infer_sketch_pose.py --img_path <complete path for sketch image path>  --model_path <complete onnx path for 2d detection>
'''
import os
import os.path as osp
import numpy as np
import onnxruntime as ort
import time
import cv2
import argparse


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r'deployment/output/sketch_detector.onnx',
                        help='the onnx model path')
    parser.add_argument('--img_path', type=str, default=r'deployment/data/test1.png', help='the test img path')
    args = parser.parse_args()
    return args


SKETCH_CLASS_INDEX = 0


class SketchDetector():
    single_threshold = 0.2
    nms_threshold = 0.3

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.output_tesnor_names = [node.name for node in self.session.get_outputs()]
        self.input_tensor_names = self.session.get_inputs()

    def preprocess(self, img_data):
        '''

        Args:
            img_data: [H,W,C]

        Returns:
            img_tensor: [N=1,C,H,W] with Normalize and ToTensor()
        '''
        norm_mean = [103.53, 116.28, 123.675]
        norm_std = [1.0, 1.0, 1.0]
        to_rgb = False
        img_tensor = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        # pad to mod by 32 for both width and height
        h, w = img_tensor.shape[:2]
        h, w = [int((_ * 1.5) // 32 * 32) for _ in [h, w]]
        h, w = min(1344, h), min(1344, w)
        # normalize
        img_tensor = (img_tensor - norm_mean) / norm_std
        # [H,W,C] -> [C,H,W]
        img_tensor = np.transpose(img_tensor, (2, 0, 1)).astype(np.float32)

        # For mmdet MaskRCNN, this operation doesn't support
        # data = data.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return img_tensor

    def inference(self, img_path, show_res=False):
        '''

        Args:
            img_path: str, path for image
            show_res: bool, if True, the result will be drawn with initial image data.

        Returns:
            combine_res: [ dict{
                              bbox: [x1,y1,x2,y2],
                              mask: [H,W], numpy.array in binary and gray
                            }
                      ]
            Attention: The results have sorted by NMS algorithms through scores of bboxes. The higher the first.
        '''
        assert osp.exists(img_path), 'Image path should exist!!'
        self.img_path = img_path
        img_data = cv2.imread(img_path)
        self.img_data = img_data
        img_tensor = self.preprocess(img_data)
        detector_results = self.session.run(self.output_tesnor_names,
                                            input_feed={self.input_tensor_names[0].name: img_tensor})

        bbox_list, label_list, mask_list = detector_results[0][0], detector_results[1][0], detector_results[2][0]

        combined_res_list = self.postprocess(img_data, bbox_list, label_list, mask_list, show_res=show_res)
        return combined_res_list

    def postprocess(self, img_data, bboxes_list, labels_list, masks_list, show_res: bool = False):
        '''

        Args:
            img_data: [H,W,C]
            bboxes_list: [num_det,5=(x1,y1,x2,y2,score)]
            labels_list: [num_det]
            masks_list: [num_det,H,W]
            show_res: bool, default False. If True, will show results for masks and detected rectangles.

        Returns:
            combined_res: [ dict{
                              bbox: [x1,y1,x2,y2],
                              mask: [H,W], numpy.array in binary and gray
                            }
                      ]
        '''
        # single filter with merely calss and threshold constraint
        single_filtered_bboxes_list, single_filtered_labels_list, single_filtered_masks_list = self._single_thresh_filter(
            bboxes_list, labels_list, masks_list,
            score_thr=SketchDetector.single_threshold)

        single_filtered_scores_list = single_filtered_bboxes_list[:, -1].reshape(-1)
        single_filtered_bboxes_list = single_filtered_bboxes_list[:, :-1]

        # nms filter
        nms_indicies = self._nms(single_filtered_bboxes_list, single_filtered_scores_list, iou_threshold=0.3)

        nms_filtered_bboxes, nms_filtered_masks, nms_filtered_labels = single_filtered_bboxes_list[nms_indicies], \
                                                                       single_filtered_masks_list[nms_indicies], \
                                                                       single_filtered_labels_list[nms_indicies]

        if show_res:
            self.vis_result(img_data, nms_filtered_bboxes, nms_filtered_masks)

        self.combined_res = self._save_bbox_and_mask_res(nms_filtered_bboxes, nms_filtered_masks, store_mask=True)

        return self.combined_res

    def _single_thresh_filter(self, bbox_list: list, label_list: list, mask_list: list, score_thr=0.3):
        '''

        Args:
            bbox_list: [num_det,5=(x1,y1,x2,y2,score)]
            label_list: [num_det]
            mask_list: [num_dete,H,W]

        Returns:
            filtered with certain threshold score where are score > score threshold

            bboxes: [filtered_num_det,5=[x1,y1,x2,y2,score]]
            labels: [filtered_num_det]
            masks: [filtered_num_dete,H,W]
        '''

        needindexs = np.where((bbox_list[:, -1] > score_thr) & (label_list == SKETCH_CLASS_INDEX))
        bboxes = bbox_list[needindexs]
        labels = label_list[needindexs]
        masks = mask_list[needindexs]

        return bboxes, labels, masks

    def _nms(self, boxes: np.array, scores: np.array, iou_threshold: float):
        '''
        NMS algorithm to filter some rectangle lower than constraint threshold.
        Args:
            boxes: [num_det,4=[x1,y1,x2,y2] ]
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

    def _save_bbox_and_mask_res(self, bbox_list: list, mask_list: list, store_mask: bool = False):
        '''
        Save detector results into a list.
        Args:
            bbox_list:  [num_det,4=[x,y,x,y]]
            mask_list: [num_det,H,W]
        Returns:
            res_list: [ dict{
                              bbox: [x1,y1,x2,y2],
                              mask: [H,W], numpy.array in binary and gray
                            }
                      ]
        '''
        if mask_list is None:
            store_mask = False
        res_list = []

        for index, (box, mask) in enumerate(zip(bbox_list, mask_list)):
            cur_res = {}
            x1, y1, x2, y2 = box.astype(np.int32)
            mask = mask * 255
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            cur_res['bbox'] = [x1, y1, x2, y2]
            if store_mask:
                cur_res['mask'] = mask

            res_list.append(cur_res)

        return res_list

    def vis_result(self, img_data, bbox_list, mask_list):
        drawn_img_data = img_data
        for index, (box, mask) in enumerate(zip(bbox_list, mask_list)):
            x1, y1, x2, y2 = box.astype(np.int32)
            mask = mask * 255
            ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            black = np.zeros_like(drawn_img_data)
            black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
            black = mask.astype(np.uint8)

            _, contours, hierarchy = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 1:
                contours = np.array(contours).reshape((-1, 1, 2))
                cv2.polylines(drawn_img_data, [contours], isClosed=True, color=(0, 0, 255), thickness=3)
            else:
                for contour in contours:
                    contour = np.array(contour).reshape((-1, 1, 2))
                    cv2.polylines(drawn_img_data, [contour], isClosed=True, color=(0, 0, 255), thickness=3)

            alpha = 0.8

            mask = mask.astype(bool)
            random_colors = np.array([0, 255, 255])
            drawn_img_data[mask] = drawn_img_data[mask] * (1 - alpha) + random_colors * alpha

            cv2.rectangle(drawn_img_data, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('', drawn_img_data)
        cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    img_path = args.img_path
    sketch_estimator = SketchDetector(model_path=model_path)
    sketch_estimator.inference(img_path=img_path, show_res=True)
