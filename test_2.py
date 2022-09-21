import os
import cv2
import time
import argparse

import torch
import model.detector
import utils.utils

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='',
                        help='The path of the .pth model to be transformed')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    # sets the module in eval node
    model.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    while (True):
        # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
        hx, frame = cap.read()
        # 如果hx为Flase表示开启摄像头失败，那么就输出"read vido error"并退出程序
        if hx is False:
            # 打印报错
            print('read video error')
            # 退出程序
            exit(0)
    # 数据预处理
        ori_img = frame
    #cv2.imread(opt.img)
        res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 模型推理
        #start = time.perf_counter()
        preds = model(img)
        #end = time.perf_counter()
        #time = (end - start) * 1000.
        #print("forward time:%fms" % time)

        # 特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

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

        #cv2.imwrite("test_result.png", ori_img)
        cv2.imshow('video', frame)
        # 监测键盘输入是否为q，为q则退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break

    # 释放摄像头
    cap.release()

    # 结束所有窗口
    cv2.destroyAllWindows()

