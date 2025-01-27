import cv2
import numpy as np

# 创建一个空白的黑色图像
image_size = (400, 400)
image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

# 定义多组线段的顶点坐标
line_segments = [
    [(100, 150), (200, 50)],
    [(300, 50), (400, 150)],
    [(300, 150), (200, 250)],
    [(100, 250), (200, 150)],
    [(300, 50), (300, 150)],
    [(100, 250), (200, 350)],
    [(300, 250), (300, 350)]
]

# 连接线段形成封闭多边形
connected_segments = []
for segment in line_segments:
    connected_segments.extend(segment)

# 设置填充颜色，这里使用白色 (255, 255, 255)
fill_color = (255, 255, 255)

# 将直接能组成多边形的线段填充
for segment in line_segments:
    if segment[-1] == line_segments[line_segments.index(segment)-1][0]:
        points = np.array(segment, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [points], fill_color)

# 将需要组合才能形成封闭区域的线段填充
combined_lines = [
    [(200, 150), (300, 50), (300, 150), (200, 250)]
]
for segment in combined_lines:
    points = np.array(segment, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [points], fill_color)

# 显示图像
cv2.imshow('Filled Regions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
