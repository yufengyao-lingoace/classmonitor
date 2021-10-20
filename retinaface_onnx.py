import onnxruntime
import numpy as np
import os
from itertools import product as product
from math import ceil


class Anchors(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape((-1, 4))
        return output


class Retinaface_onnx:
    def __init__(self):
        export_onnx_file = os.path.join(os.path.dirname(__file__), 'weights/retinaface_mobilenet.onnx')  # 输出的ONNX文件名
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 3  # 设置线程数
        opts.inter_op_num_threads = 3  # 设置线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.onnx_session = onnxruntime.InferenceSession(export_onnx_file, opts)
        self.cfg_mnet = {
            'name': 'mobilenet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'train_image_size': 840,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        }

    def detect(self, frame):
        backbone = "mobilenet"
        if backbone == "mobilenet":
            cfg = self.cfg_mnet
        else:
            cfg = self.cfg_re50
        image = np.array(frame, np.float32)
        image -= np.array((104, 117, 123), np.float32)

        inputs = np.transpose(image, (2, 0, 1))
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs.astype(np.float32)
        inputs = {self.onnx_session.get_inputs()[0].name: inputs}
        
        pred = self.onnx_session.run(None, inputs, None)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                               np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        im_height, im_width, _ = np.shape(image)
        anchors = Anchors(cfg, image_size=(im_height, im_width)).get_anchors()

        loc, conf, landms = pred[0], pred[1], pred[2]

        boxes = self.decode_np(np.squeeze(loc), anchors, cfg['variance'])
        conf = np.squeeze(conf)[:, 1:2]

        landms = self.decode_landm(np.squeeze(landms), anchors, cfg['variance'])
        boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
        boxes_conf_landms = self.non_max_suppression(boxes_conf_landms, 0.5)

        if len(boxes_conf_landms) == 0:
            return []
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4]*scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:]*scale_for_landmarks
        return boxes_conf_landms

    def non_max_suppression(self, boxes, conf_thres=0.5, nms_thres=0.3):
        detection = boxes
        # 1、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
        mask = detection[:, 4] >= conf_thres
        detection = detection[mask]
        if not np.shape(detection)[0]:
            return []

        best_box = []
        scores = detection[:, 4]
        # 2、根据得分对框进行从大到小排序。
        arg_sort = np.argsort(scores)[::-1]
        detection = detection[arg_sort]

        while np.shape(detection)[0] > 0:
            # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
            best_box.append(detection[0])
            if len(detection) == 1:
                break
            ious = self.iou(best_box[-1], detection[1:])
            detection = detection[1:][ious < nms_thres]

        return np.array(best_box)

    def iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
            np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)

        iou = inter_area/np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
        return iou

    def decode_np(self, loc, priors, variances):
        # 中心解码，宽高解码
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances):
        # 关键点解码
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                                 ), axis=1)
        return landms
