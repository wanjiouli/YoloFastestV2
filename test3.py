# import netron
# modelPath = "./yolo-fastestv2.onnx"
# netron.start(modelPath)
# import onnx
#
# onnx_model = onnx.load("./yolo-fastestv2.onnx")
# try:
#     onnx.checker.check_model(onnx_model)
# except Exception:
#     print("Model incorrect")
# else:
#     print("Model correct")
import torch
import onnxruntime
import cv2
import os
import torch.nn.functional as F
import time
import argparse

import torch
import model.detector
import utils.utils
import numpy as np
def softmax(x, axis=1):
	# 计算每行的最大值
	row_max = x.max(axis=axis)
	# 每行的元素都要减去对应行的最大值，解决溢出问题
	row_max = row_max.reshape(-1, 1)
	hatx = x - row_max
	# 计算分子
	hatx_exp = np.exp(hatx)
	# 计算分母
	hatx_sum = np.sum(hatx_exp, axis=axis, keepdims=True)
	# 计算softmax值
	s = hatx_exp / hatx_sum
	return s

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def make_grid(h, w, cfg, device):
    hv,wv = np.meshgrid(np.arange(h),np.arange(w))
    print(hv,wv)
    #hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])

    return (np.tile(np.stack([wv,hv],axis=2),(1,1,3))).reshape(h,w,cfg["anchor_num"],-1)
    # torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["anchor_num"], -1).to(device)
def handel_preds(preds, cfg, device):
    # 加载anchor配置
    anchors = np.array(cfg["anchors"])
    #anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)).to(device)
    anchors = anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)

    output_bboxes = []
    layer_index = [0, 0, 0, 1, 1, 1]

    for i in range(len(preds) // 3):
        bacth_bboxes = []

        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]
        # reg_preds = torch.from_numpy(preds[i * 3])
        # obj_preds = torch.from_numpy(preds[(i * 3) + 1])
        # cls_preds = torch.from_numpy(preds[(i * 3) + 2])
        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            r = r.transpose(1, 2, 0)
            #r = r.permute(1, 2, 0)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = o.transpose(1, 2, 0)
            #o = o.permute(1, 2, 0)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            #c = c.permute(1, 2, 0)
            c = c.transpose(1, 2, 0)
            c = c.reshape(c.shape[0], c.shape[1], 1, c.shape[2])
            c = np.tile(c,(1,1,3,1))
            #c = c.repeat(1, 1, 3, 1)
            anchor_boxes = np.zeros([r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1])
            #anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            # 计算anchor box的cx, cy
            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["height"] / r.shape[0]
            anchor_boxes[:, :, :, :2] = ((sigmoid(r[:, :, :, :2]) * 2. - 0.5) + grid) * stride
            #anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

            # 计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (sigmoid(r[:, :, :, 2:4]) * 2) ** 2 * anchors_cfg  # wh
            #anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg  # wh

            # 计算obj分数
            anchor_boxes[:, :, :, 4] = sigmoid(o[:, :, :, 0])
            #anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

            # 计算cls分数
            anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim=3)

            # torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy()
            bacth_bboxes.append(anchor_boxes)

            # n, anchor num, h, w, box => n, (anchor num*h*w), box
        bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1])

        output_bboxes.append(bacth_bboxes)

        # merge
    output = torch.cat(output_bboxes, 1)

    return output
if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    # parser.add_argument('--weights', type=str, default='',
    #                     help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='',
                        help='The path of test image')
    ort_session = onnxruntime.InferenceSession("./yolo-fastestv2.onnx")
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    #assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3).astype('float32')



    img = (img.transpose(0,3,1,2) / 255.0)

    #print(img.dtype)
    # img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    #
    # img = img.to(device).float() / 255.0

    input_name = ort_session.get_inputs()
    out_name_ = ort_session.get_outputs()

    out_name = []
    for i in out_name_:
        out_name.append(i.name)
    # print(out_name)
    # print(img.shape)
    #print({"input": img})
    #print(to_numpy(img))

    # print(img,type(img))
    #print(img_1)
    preds = ort_session.run(out_name,{input_name[0].name:img})
    #print(preds)
    #print({"input": img})

    #print(type(preds))

    #特征图后处理
    output = handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    # 绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("test_result.png", ori_img)
