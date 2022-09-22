# import netron
# modelPath = "yolo-fastestv2.onnx"
# netron.start(modelPath)
import cv2
import torch
# # x1
# x1 = torch.tensor([[11,21,31],[21,31,41]],dtype=torch.int)
# print(x1)
# print(x1.shape) # torch.Size([2, 3])
# # x2
# x2 = torch.tensor([[12,22,32],[22,32,42]],dtype=torch.int)
# print(x2)
# print(x2.shape)  # torch.Size([2, 3])
#
# inputs = [x1, x2]
# print(inputs)
# print('**********************************')
# input_1 = torch.cat(inputs, dim=0)
# print(input_1)
# input_2 = torch.cat(inputs, dim=1)
# print(input_2)
# # 假设是时间步T1的输出
# T1 = torch.tensor([[1, 2, 3],
#         		[4, 5, 6],
#         		[7, 8, 9]])
# # 假设是时间步T2的输出
# T2 = torch.tensor([[10, 20, 30],
#         		[40, 50, 60],
#         		[70, 80, 90]])
#
# print(T1)
# print(T2)
#
# print(torch.stack((T1,T2),dim=0),torch.stack((T1,T2),dim=0).shape)
# print(torch.stack((T1,T2),dim=1),torch.stack((T1,T2),dim=1).shape)
# print(torch.stack((T1,T2),dim=2),torch.stack((T1,T2),dim=2).shape)
# # print(torch.stack((T1,T2),dim=3),torch.stack((T1,T2),dim=3).shape)
#
# import torch
# a= torch.arange(30).reshape(5,6)
# print(a,a.shape)
# print(a[:,:,None],a[:,:,None].shape)
# # print('b:',a.repeat(1,2))
# # print('c:',a.repeat(2,1,1))
# at = torch.arange(3).float().view(3, 1).repeat(1, 318)
# print(at[:,:,None],at[:,:,None].shape)



# gain = torch.rand([3,374,2])
# a = torch.tensor([1,1,22,22,22,22,1])
# print(gain)
# # print(gain[:,None],gain[:,None].shape)
# print('*'*100)
# #print(gain * a)
# print(torch.max(gain, 1. / gain))
# print('*'*100)
# print(torch.max(gain, 1. / gain).max(2))
# print('*'*100)
# print(torch.max(gain, 1. / gain).max(2)[0])
# #j = torch.max(gain, 1. / gain).max(2)[0] < 2
# print('*'*100)
# print(torch.max(gain, 1. / gain).max(2)[0] < 2)
# g = 0.5
# gxy = torch.tensor([
#         [10.2960, 12.3544],
#         [18.7880,  6.4240],
#         [ 5.9840, 12.6331],
#         [ 1.9800,  8.5556],
#         [ 2.3100,  7.6389],
#         [ 5.1480, 11.7944],
#         [10.4060, 16.0111],
#         [15.8620,  8.0667],
#         [14.2560, 12.3444],
#         [ 1.9140, 11.0902],
#         [ 3.9160,  9.9180],
#         [ 6.8640,  9.8880],
#         [ 8.8440,  9.7978],
#         [10.8680, 15.0773],
#         [12.4300, 15.1947],
# ])
# print(gxy)
# print('*'*100)
# #j, k = ((gxy % 1. < g) & (gxy > 1.)).T
# print(gxy % 1.< g)
# print('*'*100)
# print(gxy > 1.)
# print('*'*100)
#
# print(((gxy % 1. < g) & (gxy > 1.)))
# print('*'*100)
# print(((gxy % 1. < g) & (gxy > 1.)).T)


import numpy as np
import torch
from numpy import *

# array = np.array([1, 2, 3, 4, 5, 6, ])
# print(array,type(array))
# print(torch.from_numpy(array))

# a=array([10,20])
# b = tile(a,(1,2))
# print(b)
# from numpy import random
# a = np.random.randint(5, size=(4,4,3,4))
#
# print(a)
# def softmax(x):
#     x_row_max = x.max(axis=-1)
#     x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
#     x = x - x_row_max
#     x_exp = np.exp(x)
#     x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
#     softmax = x_exp / x_exp_row_sum
#     return softmax
# print(softmax(a))
# print(softmax(a).shape)

import numpy as np
from numpy import array


def box_area(boxes: array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1: array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
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


def numpy_nms(boxes: array, scores: array, iou_threshold: float):
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


box = np.array([[2, 3.1, 7, 5], [3, 4, 8, 4.8], [4, 4, 5.6, 7], [0.1, 0, 8, 1]])
score = np.array([0.5, 0.3, 0.2, 0.4])

output = numpy_nms(boxes=box, scores=score, iou_threshold=0.3)
print(output)

cv2.dnn.NMSBoxes()