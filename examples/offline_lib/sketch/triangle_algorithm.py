import os
import os.path as osp
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay
from skimage import measure
from shapely import geometry
import cv2
from typing import List, Tuple, Union
import matplotlib.pyplot as plt


def padding_img_to_rectangle(img: npt.NDArray[np.uint8]):
    '''
    将图片填充为长宽一致，是一个方形图片
    '''
    h, w = img.shape[0], img.shape[1]
    rect_l = max(h, w)
    if len(img.shape) == 3:
        rect_img = np.zeros((rect_l, rect_l, 3), dtype=np.uint8)
    else:
        rect_img = np.zeros((rect_l, rect_l), dtype=np.uint8)
    rect_img[:h, :w, ...] = img
    # cv2.imshow('tmp', rect_img)
    # cv2.waitKeyEx(-1)
    return rect_img


# 利用 Delaunay 实现三角面片化
def generate_triangle_mesh(mask: npt.NDArray[np.uint8]):
    '''
    :param  mask    binarized image [H,W,C],
                    Please attention: in actually, the coordinate system is [y,x] but in triangle the right order is [x,y],
                    Therefore, we should rotate 90 in counterclockwise orientation at first.
    :return  dict{
                'vertices': all points include contours and inside vertices,
                'triangles_idxs': [V,C=3] , V express there are V inside triangle's indicies whoes centroid is in contour
                },
    在这里请注意y轴的方向是垂直向上(这和图像的坐标朝向是不一样的)，x的方向是水平向右的
    '''
    # 1.要求输入的对图片做旋转操作
    # 关于这个旋转的解释，暂时参考这个 https://github.com/facebookresearch/AnimatedDrawings/issues/220
    mask = np.rot90(mask, 3, )  # rotate origin point to upright
    # cv2.imshow('tmp', mask)
    # cv2.waitKey(-1)
    # 2.补充图片为一个正方形
    padded_img: npt.NDArray[np.uint8] = padding_img_to_rectangle(mask)
    h, w = padded_img.shape[0], padded_img.shape[1]
    # 3.先找到大致的多边形集合
    contours: List[npt.NDArray[np.float64]] = measure.find_contours(padded_img, 128)
    if len(contours) > 1:
        # print('Contours more than one!!')
        # contours = contours.sort(key=len, reverse=True)
        assert False, 'Contours more than one!!'
    # 4.对多边形轮廓做精简，最终得到的是外层的点
    outside_verticies: npt.NDArray[np.float64] = measure.approximate_polygon(contours[0], tolerance=0.25)
    mask_outline = geometry.Polygon(contours[0])

    # 5.set sample points and defines sample pointset in the range of polygon
    inside_vertices_xy: List[Tuple[np.float32, np.float32]] = []
    _x = np.linspace(0, w, 40)
    _y = np.linspace(0, h, 40)
    # test meshgrid:
    #     a = [1, 2, 3]
    #     b = [4, 5, 6]
    #     cx, cy = np.meshgrid(a, b)
    # result grouped by cx and cy likes : [[1,4],[2,4],[3,4],
    #                                      [1,5],[2,5],[3,5],
    #                                      [1,6],[2,6],[3,6]]
    xv, yv = np.meshgrid(_x, _y)
    for x, y in zip(xv.flatten(), yv.flatten()):
        if mask_outline.contains(geometry.Point(x, y)):
            inside_vertices_xy.append((x, y))
    inside_vertices: npt.NDArray[np.float64] = np.array(inside_vertices_xy)

    # 6.final points set = contour points set + inside points set , number is N
    # [N,2]
    vertices: npt.NDArray[np.float64] = np.concatenate([outside_verticies, inside_vertices]).astype(np.float32)

    # 7.excute Delaunay triangle algorithm with given final points set
    convex_hull_triangles = Delaunay(vertices)
    # [V,3], suppose there have V triangle , eaach triangle has 3 indicies
    triangles_idx_tuple: List[npt.NDArray[np.int32]] = []
    # 8.Excluding triangles where the centroid of the triangle face is not within the mask contour
    for _triangle in convex_hull_triangles.simplices:
        # verticies of a triangle [(x1,y1),(x2,y2),(x3,y3)]
        # [index cnt=3,C=2]
        tri_vertices = np.array([
            vertices[_triangle[0]], vertices[_triangle[1]], vertices[_triangle[2]]
        ])
        # caculate Triangle centroid = [ (x1+x2+x3)/3 , (y1+y2+y3)/3 ]
        tri_centroid = geometry.Point(np.mean(tri_vertices, axis=0))
        if mask_outline.contains(tri_centroid):
            triangles_idx_tuple.append(_triangle)

    # vertices[:, 0] /= w
    # vertices[:, 1] /= h
    res = {'vertices': vertices, 'triangles_idxs': triangles_idx_tuple}
    return res


def plot_mesh_plt(vertices, triangles_idxs, pins_xy, x_axis_range=(-15, 15), y_axis_range=(-15, 15),
                  show_res=True,
                  save_data=False, save_base_dir=''):
    """ Helper function to visualize mesh deformation outputs """
    for tri in triangles_idxs:
        x_points = []
        y_points = []
        v0, v1, v2 = tri.tolist()
        x_points.append(vertices[v0][0])
        y_points.append(vertices[v0][1])
        x_points.append(vertices[v1][0])
        y_points.append(vertices[v1][1])
        x_points.append(vertices[v2][0])
        y_points.append(vertices[v2][1])
        x_points.append(vertices[v0][0])
        y_points.append(vertices[v0][1])

        plt.plot(x_points, y_points)
    plt.xlim(x_axis_range)
    plt.ylim(y_axis_range)
    plt.ylabel('y')

    for pin in pins_xy:
        plt.plot(pin[0], pin[1], color='red', marker='o')

    if show_res:
        plt.show()

    if save_data:
        plt.savefig(osp.join(save_base_dir, 'triangle_image.png'))


def plot_mesh_img(src: npt.NDArray[np.uint8], vertices: npt.NDArray[np.float64],
                  triangles_idxs: List[npt.NDArray[np.int32]],
                  pins_xy: npt.NDArray[np.float64],
                  need_padding_to_rect: bool = True,
                  need_visualize: bool = True):
    '''
    针对三角化后的结果做可视化，这里需要注意的是，这个三角化过程得到的结果坐标系和图像空间的坐标系是不一样的
    x正方向向右，而y的正方向向下
    '''
    src_copy = np.copy(src)
    if need_padding_to_rect:
        src_copy = padding_img_to_rectangle(src_copy)
    h, w = src_copy.shape[0], src_copy.shape[1]

    for tri in triangles_idxs:
        v0_i, v1_i, v2_i = tri.tolist()

        v0 = vertices[v0_i]
        v1 = vertices[v1_i]
        v2 = vertices[v2_i]

        cv2.circle(src_copy, (int(v0[0]), h - int(v0[1])), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(src_copy, (int(v1[0]), h - int(v1[1])), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(src_copy, (int(v2[0]), h - int(v2[1])), radius=3, color=(0, 0, 255), thickness=-1)

        cv2.line(src_copy, pt1=(int(v0[0]), h - int(v0[1])),
                 pt2=(int(v1[0]), h - int(v1[1])),
                 color=(255, 0, 0), thickness=2)
        cv2.line(src_copy, pt1=(int(v1[0]), h - int(v1[1])),
                 pt2=(int(v2[0]), h - int(v2[1])),
                 color=(255, 0, 0), thickness=2)
        cv2.line(src_copy, pt1=(int(v2[0]), h - int(v2[1])),
                 pt2=(int(v0[0]), h - int(v0[1])),
                 color=(255, 0, 0), thickness=2)

        # cv2.circle(src_copy, (int(v0[0]), int(v0[1])), radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.circle(src_copy, (int(v1[0]), int(v1[1])), radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.circle(src_copy, (int(v2[0]), int(v2[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #
        # cv2.line(src_copy, pt1=(int(v0[0]), int(v0[1])),
        #          pt2=(int(v1[0]), int(v1[1])),
        #          color=(255, 0, 0), thickness=2)
        # cv2.line(src_copy, pt1=(int(v1[0]), int(v1[1])),
        #          pt2=(int(v2[0]), int(v2[1])),
        #          color=(255, 0, 0), thickness=2)
        # cv2.line(src_copy, pt1=(int(v2[0]), int(v2[1])),
        #          pt2=(int(v0[0]), int(v0[1])),
        #          color=(255, 0, 0), thickness=2)

    for pin in pins_xy:
        cv2.circle(src_copy, (int(pin[0]), h - int(pin[1])), radius=3, color=(0, 0, 255), thickness=-1)

    if need_visualize:
        if len(src_copy.shape) == 2:
            canvas = cv2.cvtColor(src_copy, cv2.COLOR_GRAY2BGR)
        else:
            canvas = src_copy.copy()
        cv2.imshow('res', canvas)
        cv2.waitKeyEx(-1)

    return src_copy


if __name__ == '__main__':
    mask_path = osp.join('test_data', 'triangle_character', 'mask.png')
    # 输入灰度图，并作二值化
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    ret, mask_copy = cv2.threshold(mask.copy(), 0, 255, cv2.THRESH_OTSU)

    # 做三角形网格划分
    res = generate_triangle_mesh(mask_copy)
    # 用matplotlib做可视化
    plot_mesh_plt(vertices=res['vertices'], triangles_idxs=res['triangles_idxs'], pins_xy=np.array([]),
                  x_axis_range=(0, mask.shape[1]), y_axis_range=(0, mask.shape[0]))
    # 用opencv做可视化
    plot_mesh_img(src=mask.copy(), vertices=res['vertices'], triangles_idxs=res['triangles_idxs'], pins_xy=np.array([]))
