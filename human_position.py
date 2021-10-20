import numpy as np
import cv2
from scipy.spatial import distance as dist

resize_fold = 1.7
TEACHER_WIDTH = 360
TEACHER_HEIGHT = 360
t_X_pos = 0
t_Y_pos = 0

def contour_position_change(contours, img_width = 0, img_height = 0):
    for d1 in contours:
        for d2 in d1:
            for d3 in d2:
                d3[0] = d3[0] + img_width
                d3[1] = d3[1] + img_height
def contour_bound_detect(contours):
    max_y = -5000
    min_y = 5000
    for d1 in contours:
        for d2 in d1:
            for d3 in d2:
                if d3[1] > max_y:
                    max_y = d3[1]
                if d3[1] < min_y:
                    min_y = d3[1]
    return max_y, min_y

def position_logic(points_in_contour, point_pos, bound):
    out_problem = [0, 0, 0, 0]
    # 距离过近
    if points_in_contour[0] < 0 and points_in_contour[16] < 0 and  points_in_contour[8] > 0:
        out_problem[0] = 1
    # 人像水平偏移
    if points_in_contour[19] * points_in_contour[24] < 0 or \
        (points_in_contour[19] < 0 and points_in_contour[24] < 0 and points_in_contour[33] < 0):
        out_problem[1] = 1
    face_size_idx = dist.euclidean(point_pos[19], point_pos[24])
    # 距离过远
    if face_size_idx < 65*240/360:
        out_problem[2] = 1
    head_height_percent = (point_pos[30][1] - bound[1]) / (bound[0] - bound[1])
    # 人像竖直偏移
    if head_height_percent < 0.2 or head_height_percent >= 0.5:
        out_problem[3] = 1
    return out_problem

def teacher_window_collect(frame):
    #frame = cv2.resize(frame, (1920, 1080))
    frame = frame[t_Y_pos:t_Y_pos+TEACHER_HEIGHT, t_X_pos:t_X_pos+TEACHER_WIDTH]
    return frame
def process_frame_figure(f_name):
    pos_img = cv2.imread(f_name, 1)
    pos_img = cv2.resize(pos_img, None, fx=resize_fold,fy=resize_fold,interpolation=cv2.INTER_CUBIC)
    p_height, p_width, _ = pos_img.shape
    ret, thresh = cv2.threshold(cv2.cvtColor(pos_img.copy(), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_position_change(contours, (TEACHER_WIDTH - p_width) / 2 , TEACHER_HEIGHT - p_height)
    y_max, y_min = contour_bound_detect(contours)
    return contours, y_max, y_min

def positon_check0(face_features, contours, y_max, y_min):
    far_ts = []
    close_ts = []
    vertical_ts = []
    hori_ts = []
    for idx, faces, img, ts in face_features:
        for face in faces:
            d_points = {}
            points_pos = {}
            landmarks = np.matrix([[p.x, p.y] for p in face.parts()])
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                d_points[idx] = cv2.pointPolygonTest(contours[0], pos, True)
                points_pos[idx] = pos
            problems = position_logic(d_points, points_pos, (y_max, y_min))
            if problems[0] == 1:
                close_ts.append(ts)
            if problems[1] == 1:
                hori_ts.append(ts)
            if problems[2] == 1:
                far_ts.append(ts)
            if problems[3] == 1:
                vertical_ts.append(ts)
    return far_ts, close_ts, vertical_ts, hori_ts

def positon_check(landmarks, contours, y_max, y_min):
    d_points = {}
    points_pos = {}
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        d_points[idx] = cv2.pointPolygonTest(contours[0], pos, True)
        points_pos[idx] = pos
    problems = position_logic(d_points, points_pos, (y_max, y_min))
    return problems


