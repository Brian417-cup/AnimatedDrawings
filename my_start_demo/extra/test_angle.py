import numpy as np
import numpy.typing as npt
import math

# 这边的操作是为了证明能快速计算两个向量的夹角方法（带方向的，不用余弦而用行列式的方法来做）
# 设v1(x1,y1) v2(x2,y2)
# 则 angle= arctan(det=(v1,v2),dot=(v1,v2))
# 这样求出来的angle是从v1- > v2 的带方向夹角

v1 = np.array([[1.0, 0.0], [2, 2]], dtype=np.float32)
v2 = np.array([[2.0, 2.0 * math.sqrt(3)], [1.0, 0.0]], dtype=np.float32)

det: npt.NDArray[np.float32] = v1[..., 0] * v2[..., 1] - v2[..., 0] * v1[..., 1]
dot: npt.NDArray[np.float32] = v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]
angle = np.arctan2(det, dot)
print(angle, [math.pi / 3, -math.pi / 4])

sum = 0.54189
cur_len = np.linalg.norm(np.array([0.0437, 0.2698, 0.0019], dtype=np.float32), ord=2)
sum += cur_len
print(sum)

a = np.array([1, 2, 3], dtype=np.float32)
v1 = a[::-1] * np.array([-1, 1, -1])
print(v1)

a = np.array([[0.3846154, 0.7948718]])
b = np.array([[0.3846154, 0.82051283]])
c = np.array([[0.35897437, 0.7948718]])
p = np.array([[0.38372093, 0.8023256]])

ab = np.linalg.norm((a - b), ord=2, axis=-1)
ac = np.linalg.norm((a - c), ord=2, axis=-1)
bc = np.linalg.norm((c - b), ord=2, axis=-1)


def caculate_s(ab, ac, bc):
    helen_p = (ab + ac + bc) / 2
    s = np.sqrt(helen_p * (helen_p - ab) * (helen_p - ac) * (helen_p - bc))
    return s


total_s = caculate_s(ab, ac, bc)

pb = np.linalg.norm((p - b), ord=2, axis=-1)
pc = np.linalg.norm((p - c), ord=2, axis=-1)
pa = np.linalg.norm((p - a), ord=2, axis=-1)

s_abp = caculate_s(ab, pa, pb)
s_acp = caculate_s(ac, pa, pc)
s_bcp = caculate_s(bc, pb, pc)

u = s_abp / total_s
v = s_acp / total_s
w = s_bcp / total_s
print(u, v, w)
print(np.sum([u,v,w]))
