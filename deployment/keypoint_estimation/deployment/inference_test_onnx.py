# reffered from：https://blog.csdn.net/LSC_333/article/details/117337137

# Execute following command:
# python inference_test_onnx.py --model_path your/onnx/model/path --img_path your/img/complete/path
import onnxruntime
import cv2
import os.path as osp
import numpy as np
from top_down_eval import keypoints_from_heatmaps
import argparse

model_path = osp.join('output', 'human_skeleton.onnx')
img_path = osp.join('', 'data', 'test1.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=f'{model_path}', help='the onnx model path')
    parser.add_argument('--img_path', type=str, default=f'{img_path}', help='the test img path')
    args = parser.parse_args()
    return args


def inference_with_onnx(model_path: str, img_path: str):
    session = onnxruntime.InferenceSession(model_path)
    output_tensor_names = [node.name for node in session.get_outputs()]
    input_tensor_names = session.get_inputs()
    img_data = cv2.imread(img_path)
    img_tensor = cv2.dnn.blobFromImage(img_data, scalefactor=1.0 / 255, size=(192, 256), mean=[0.485, 0.456, 0.406],
                                       swapRB=True,
                                       crop=False)
    output_res = session.run(output_tensor_names, input_feed={input_tensor_names[0].name: img_tensor})
    return output_res


def vis_pose(img, points):
    for i, point in enumerate(points):
        x, y = point
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img, '{}'.format(i), (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    args = get_args()

    heatmap = inference_with_onnx(model_path=args.model_path, img_path=args.img_path)[0]
    # print(heatmap.shape)
    img_data = cv2.imread(args.img_path)
    h, w, c = img_data.shape
    # 代表目标检测框的左上角顶点
    x, y = 0, 0
    center = np.array([[x + w * 0.5, y + h * 0.5]], dtype=np.float32)

    scale = np.array([[w / 200, h / 200]], dtype=np.float32)
    res = keypoints_from_heatmaps(heatmap, center, scale)[0]
    res = res.tolist()
    print(res)
    img = vis_pose(img_data, res[0])
    cv2.imshow('result', img)
    cv2.waitKey(-1)
