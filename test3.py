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
import torchvision

def bbox_iou(box1,box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0],box1[1],box1[2],box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    inter_rect_x1 = np.maximum(b1_x1,b2_x1)
    inter_rect_y1 = np.maximum(b1_y1,b2_y1)
    inter_rect_x2 = np.minimum(b1_x2,b2_x2)
    inter_rect_y2 = np.minimum(b1_y2,b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1,0) * np.maximum(inter_rect_y2 - inter_rect_y1,0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y2)
    area_b2 = (b2_x2 - b2_x1) * (b1_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

def non_max_suppression_1(boxes,sigma=0.5,conf_thres=0.5,nms_thres=0.4):
    # boxes (n,anchor num,4+1+80)
    bs = np.shape(boxes)[0]       # 提取batch_size
    # 将框转换成左上角右下角的形式
    shape_boxes = np.zeros_like(boxes[:,:,:4])
    shape_boxes[:,:,0] = boxes[:,:,0] - boxes[:,:,2] / 2
    shape_boxes[:,:,1] = boxes[:,:,1] - boxes[:,:,3] / 2
    shape_boxes[:,:,2] = boxes[:,:,0] + boxes[:,:,2] / 2
    shape_boxes[:,:,3] = boxes[:,:,1] + boxes[:,:,3] / 2

    boxes[:,:,:4] = shape_boxes
    output = []
    # 按batch数量循环
    for i in range(bs):
        # 提取第i batch的预测结果
        prediction = boxes[i]
        # 提取预测的第5列的的数值,这个数值代表是否检测出物体
        score = prediction[:,4]
        # 过滤掉小于conf_thres设置的值的框框,这一步将极大的筛选掉框框的数量
        mask = score > conf_thres
        detections = prediction[mask]
        # 提取预测的类别分数
        w = np.max(detections[:, 5:], axis=-1)
        class_conf = np.expand_dims(np.max(detections[:,5:],axis=-1),axis=-1)
        # 提取预测的种类类别
        class_pred = np.expand_dims(np.argmax(detections[:,5:],axis=-1),axis=-1)
        # 将 detections[:,:5] xyxy n,class_conf 预测类别的概率,class_pred 预测的类别 按最后一维拼接 这里就是按列拼接
        detections = np.concatenate([detections[:,:5],class_conf,class_pred],-1)
        # 去除重复的类别
        unique_class = np.unique(detections[:,-1])
        if len(unique_class) == 0:
            continue
        # 用于过渡使用
        best_box = []
        # 按类别进行nms
        for c in unique_class:
            # 将不是当前正在循环的类别排除掉
            cls_mask = detections[:,-1] == c
            detection = detections[cls_mask]
            # 提取是否物体的数值
            scores = detection[:,4]
            # 第一步是np.argsort 将按scores的值从小到大的返回index,后有让将index取反,按从从大到小排列 最后对detection进行排列
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]

            # 柔性非极大抑制
            # while np.shape(detection)[0] > 0:
            #     best_box.append(detection[0])
            #     if len(detection) == 1:
            #         break
            #     # 计算每一次有物体分数最大的best_box与其他非最大的iou
            #     ious = bbox_iou(best_box[-1],detection[1:])
            #     # 计算IOU取高斯指数后乘上原得分
            #     detection[1:,4] = np.exp(-(ious * ious) / sigma) * detection[1:,4]
            #     # 提取除第一个数据后的数据
            #     detection = detection[1:]
            #     # 保留得分大于mns_thres 删除小于的mns_thres
            #     detection = detection[detection[:, 4] >= conf_thres]
            #     # 提取当前的score
            #     scores = detection[:,4]
            #     # 第一步是np.argsort 将按scores的值从小到大的返回index,后有让将index取反,按从从大到小排列 最后对detection进行排列 再次循环
            #     arg_sort = np.argsort(scores)[::-1]
            #     detection = detection[arg_sort]

            # 普通非极大抑制
            while len(detection) != 0:
                # 将检测有物体分数最大的放入best_box中
                best_box.append(detection[0])
                # 如何detection 只有一个物体,就跳出循环
                if len(detection) == 1:
                    break
                # 计算每一次有物体分数最大的best_box与其他非最大的iou
                ious = bbox_iou(best_box[-1],detection[1:])
                # 删除大于ious的框框,保留小于ious的框框
                detection = detection[1:][ious < nms_thres]
        output.append(best_box)

    return output

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
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue
        #x[:,[0,1]] = x[:,[1,0]]

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        #conf, j = x[:, 5:].max(1, keepdim=True)
        conf = np.amax(x[:, 5:], axis=1,keepdims=True)
        j = np.argmax(x[:, 5:], axis=1,keepdims=True)
        x = np.concatenate((box,conf,j.astype('float32')),1)[conf.reshape(-1) > conf_thres]

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

        # i = numpy_nms(box,scores,iou_thres)
        # i = np.array(nms(box, scores=scores, threshold=iou_thres))

        print('*******************')

        # i = cv2.dnn.NMSBoxes(boxes.tolist(),scores, 0.3,0.4)
        # i = i.reshape(-1)
        # print(i,type(i))
        print('***************************')
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        #output[xi] = x[i].detach().cpu()
        print(i)
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
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
    xs,ys = np.meshgrid(np.arange(h),np.arange(w)) # np.meshgrid(),返回X的坐标与Y的坐标

    return (np.tile(np.stack([xs,ys], axis=2), (1, 1, 3))).reshape(h, w, cfg["anchor_num"], -1) # xs与ys在第三维上堆叠,形成对应的坐标点,后在第三维进行扩展3次，最好进行reshape为(22,22,3,2)

def handel_preds(preds, cfg):
    # 加载先验框大小配置 [12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87]
    anchors = np.array(cfg["anchors"])

    # 将先验框按顺序分为 (2, 3, 2)维度
    anchors = anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)

    # 用来存检测出来的目标框
    output_bboxes = []

    # 按特征层的数量进行循环特征后处理。这里只叙述第一次循环,后一次循环是一样的操作
    for i in range(len(preds) // 3):
        # 用于中间过度时,存检测出来的目标框
        bacth_bboxes = []

        # 提取第一层预测目标区域(batch,12,22,22)
        reg_preds = preds[i * 3]
        # 提取第一层预测目标的置信度(batch,3,22,22)
        obj_preds = preds[(i * 3) + 1]
        # 提取第一层预测目标的类别(batch,80,22,22)
        cls_preds = preds[(i * 3) + 2]

        # 将第一层的 预测目标区域,置信度,类别 进行特征后处理
        for r, o, c in zip(reg_preds, obj_preds, cls_preds): # 这里有个降维的效果
            # 调整预测区域的的维度顺序,从(12,22,22) -> (22,22,12)
            r = r.transpose(1, 2, 0)
            # 将预测区域的三维reshape为(22,22,3,4)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            # 调整置信度维度顺序 从(3,22,22) -> (22,22,3)
            o = o.transpose(1, 2, 0)
            # reshape为(22,22,3,1)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            # 调整类别维度顺序 从(80,22,22) -> (22,22,80)
            c = c.transpose(1, 2, 0)
            # reshape为(22,22,1,80)
            c = c.reshape(c.shape[0], c.shape[1], 1, c.shape[2])
            # (22,22,3,80) 扩展第三维度的数据为原来的3次
            c = np.tile(c,(1,1,3,1))

            # 设置一个维度为(22,22,3,85) 零矩阵 用于后续存储cx cy w h obj分数 cls分数
            anchor_boxes = np.zeros([r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1])

            # 计算anchor box的cx, cy
            # 获取(22,22,3,2)的网格坐标
            grid = make_grid(r.shape[0], r.shape[1], cfg)
            # 获取倍率因子
            stride = cfg["height"] / r.shape[0]
            # 将预测的目标区域的cx,cy偏移量经过sigmoid处理,再乘2,减去0.5 加上对应网格的坐标,追后乘上倍率因子,得到预测目标原图的中心坐标cx,cy
            anchor_boxes[:, :, :, :2] = ((sigmoid(r[:, :, :, :2]) * 2. - 0.5) + grid) * stride

            # 计算anchor box的w, h
            # 读取anchors的宽高
            anchors_cfg = anchors[i]
            # 计算每个anchor box 的宽高.每个点有3个anchor box,且这三个anchor box 的宽高是不同的
            anchor_boxes[:, :, :, 2:4] = (sigmoid(r[:, :, :, 2:4]) * 2) ** 2 * anchors_cfg  # wh

            # 计算类别分数 是否有目标
            anchor_boxes[:, :, :, 4] = sigmoid(o[:, :, :, 0])

            # 计算是哪一个类别的分数 softmax 对(22,22,3,80)第三维度进行计算
            anchor_boxes[:, :, :, 5:] = softmax(c[:, :, :, :])

            bacth_bboxes.append(anchor_boxes)

        bacth_bboxes = np.array(bacth_bboxes)
        # n, anchor num, h, w, box => n, (anchor num*h*w), box  (n,h,w,ahcnor num,box) => (n,anchor * h * w,box)
        bacth_bboxes = bacth_bboxes.reshape(bacth_bboxes.shape[0],-1,bacth_bboxes.shape[-1])

        output_bboxes.append(bacth_bboxes)
    # concatenate 按第二维度合并
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
    # load onnx形式的权重
    ort_session = onnxruntime.InferenceSession("./yolo-fastestv2.onnx")

    # 解析传入的参数
    opt = parser.parse_args()

    # 读取data参数
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 数据预处理
    # 读取被检测图像
    ori_img = cv2.imread(opt.img)
    # 重置图片大小
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    # 增加整图像维度数 并设置图像的像素值的类型
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3).astype('float32')
    # 调整图像的维度顺序并将图像的像素值转化为0-1之间
    img = (img.transpose(0,3,1,2) / 255.0)

    # 获取图像数据输入名称
    input_name = ort_session.get_inputs()
    # 获取输出名称
    out_name_ = ort_session.get_outputs()

    # 获取onnx的输出名称
    out_name = []
    for i in out_name_:
        out_name.append(i.name)

    start = time.perf_counter()
    preds = ort_session.run(out_name,{input_name[0].name:img})
    end = time.perf_counter()

    # 特征图后处理 得出所有的检测出的目标框
    output = handel_preds(preds, cfg)

    # 将所有检测出来目标框进行非极大值抑制进行去重
    output_boxes = non_max_suppression_1(output, conf_thres = 0.3, nms_thres = 0.4)
    # output_boxes = non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

    # 加载标签名,并保存在LABEL_NAMES列表中
    LABEL_NAMES = []

    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())


    # 获取输入图像的原始高宽
    h, w, _ = ori_img.shape

    # 获取高宽的倍率因子
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # 从列表中取出有效框 绘制预测框
    for box in output_boxes[0]:
        # 将ndarray格式转化为列表格式
        box = box.tolist()
        # 提取置信度分数
        obj_score = box[5]
        # 提取类别
        category = LABEL_NAMES[int(box[6])]
        # 计算目标框左上角点的坐标
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        # 计算目标框右下角点的坐标
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        # 在原图上画检测处的目标框
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        # 标记目标的置信度
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        # 标记目标的类别
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
    # 展示绘画后的图像
    cv2.imshow('test_result.png', ori_img)
    # 监听键盘
    cv2.waitKey(0)
    # 保存图片
    cv2.imwrite("test_result.png", ori_img)
# 关闭所有窗口
cv2.destroyAllWindows()
