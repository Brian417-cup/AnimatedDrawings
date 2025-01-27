import onnx
import onnxruntime
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import urllib.request     # urllib is a built-in Python library to download files from URLs

onnx_model_url = "https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/faster_rcnn_R_50_FPN_1x.onnx"
output_classes_url = "https://onnxzoo.blob.core.windows.net/models/opset_10/faster_rcnn/coco_classes.txt"

urllib.request.urlretrieve(onnx_model_url, filename="faster_rcnn_R_50_FPN_1x.onnx")
urllib.request.urlretrieve(output_classes_url, filename="coco_classes.txt")


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image


# Remember the coco_classes.txt file we downloaded from github earlier?
# These will provide the labels for the outputs of our ONNX model. Let's take a peek at the first 5 classes.

classes = [line.rstrip('\n') for line in open('coco_classes.txt')]
classes[0:5]

session = onnxruntime.InferenceSession('faster_rcnn_R_50_FPN_1x.onnx')

img = Image.open('images/frcnn_demo.jpg')

boxes, labels, scores = session.run(None, {
    session.get_inputs()[0].name: img_data
})

print('Boxes:', boxes.shape)
print('Labels:', labels.shape)
print('Scores:', scores.shape)

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes = boxes / ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

display_objdetect_image(img, boxes, labels, scores)