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
import torchvision
import torch
import model.detector
import utils.utils
import numpy as np
def box_area(boxes ):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(box1 , box2):
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
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes , scores , iou_threshold :float):

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

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    # 从大到小对应的的索引
    order = scores.argsort()[::-1]

    # 记录输出的bbox
    keep = []
    while order.size > 0:
        i = order[0]
        # 记录本轮最大的score对应的index
        keep.append(i)

        if order.size == 1:
            break

        # 计算当前bbox与剩余的bbox之间的IoU
        # 计算IoU需要两个bbox中最大左上角的坐标点和最小右下角的坐标点
        # 即重合区域的左上角坐标点和右下角坐标点
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 如果两个bbox之间没有重合, 那么有可能出现负值
        w = np.maximum(0.0, (xx2 - xx1))
        h = np.maximum(0.0, (yy2 - yy1))
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 删除IoU大于指定阈值的bbox(重合度高), 保留小于指定阈值的bbox
        ids = np.where(iou <= threshold)[0]
        # 因为ids表示剩余的bbox的索引长度
        # +1恢复到order的长度
        order = order[ids + 1]

    return keep



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 1000  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()

    #output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]
    output = [np.zeros((0,6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue
        x[:,[0,1]] = x[:,[1,0]]

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        #conf, j = x[:, 5:].max(1, keepdim=True)
        conf = np.amax(x[:, 5:], axis=1,keepdims=True)
        j = np.argmax(x[:, 5:], axis=1,keepdims=True)
        x = np.concatenate((box,conf,j.astype('float32')),1)[conf.reshape(-1) > conf_thres]
        #x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        # if classes is not None:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        #i = numpy_nms(box,scores,iou_thres)
        #i = np.array(nms(box, scores=scores, threshold=iou_thres))

        print('*******************')

        i = cv2.dnn.NMSBoxes(boxes.tolist(),scores, 0.3, 0.99)
        i = i.reshape(-1)
        print(i,type(i))
        print('***************************')
        #i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        #output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        # output[xi] = x

    return output

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def make_grid(h, w, cfg):
    hv,wv = np.meshgrid(np.arange(h),np.arange(w))

    #hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])

    return (np.tile(np.stack([wv,hv],axis=2),(1,1,3))).reshape(h,w,cfg["anchor_num"],-1)
    # torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["anchor_num"], -1).to(device)
def handel_preds(preds, cfg):
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
            grid = make_grid(r.shape[0], r.shape[1], cfg)
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

            anchor_boxes[:, :, :, 5:] = softmax(c[:, :, :, :])
            #anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim=3)

            # torch tensor 转为 numpy array
            #anchor_boxes = anchor_boxes.cpu().detach().numpy()
            bacth_bboxes.append(anchor_boxes)

            # n, anchor num, h, w, box => n, (anchor num*h*w), box
        #bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = np.array(bacth_bboxes)

        bacth_bboxes = bacth_bboxes.reshape(bacth_bboxes.shape[0],-1,bacth_bboxes.shape[-1])
        #bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1])

        output_bboxes.append(bacth_bboxes)

        # merge

    #output = torch.cat(output_bboxes, 1)
    output = np.concatenate(output_bboxes, 1)
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

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    start = time.perf_counter()
    preds = ort_session.run(out_name,{input_name[0].name:img})
    end = time.perf_counter()
    # spend_time = (end - start) * 1000.
    # print("forward time:%fms"%spend_time)
    #print(preds)
    #print({"input": img})

    #print(type(preds))

    #特征图后处理
    output = handel_preds(preds, cfg)
    output_boxes = non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

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
    cv_img = cv2.imread('test_result.png')
    cv2.imshow("cv_img", cv_img)
    cv2.waitKey(0)
    cv2.destroyWindow()
    spend_time = (end - start) * 1000.
    print("total time:%fms"%spend_time)