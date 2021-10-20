# @Time : 2020/7/01 18:41
# @Author : Chaochen Wu
# @File : eye_utils.py
import cv2
import dlib
import numpy as np

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in range(side[0], side[1])]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        return (cx, cy)
    except:
        return None

def check_point_in_eye(point, side, shape):
    eye_hull = cv2.convexHull(np.array(shape[side[0]:side[1]]))
    # print(eye_hull,point)
    try:
        result=cv2.pointPolygonTest(eye_hull, point, True)
        return result
    except Exception as ex:
        return False

def eyeballs_track(img,shape, threshold=105):
    '''
    通过输入的每帧人脸长方形，输出人眼睛坐标
    Args:
        face_features: img人脸特征集合。形式为[(frame_idx, faces_feature, img),...]

    Returns: 
        人眼睛坐标与人眼睛坐标与眼睛特征多边形距离列表
        [{'right_eye_center': (303, 279), 'left_eye_center': (376, 280), 'right_eye_distance': 4.0, 'left_eye_distance': 4.0}, ..]
    '''
    out_list = []
    kernel = np.ones((9, 9), np.uint8)
    
    curr_center = []
    curr_face_center = []
    FACIAL_LANDMARKS_IDXS = dict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
    ])
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = eye_on_mask(mask, FACIAL_LANDMARKS_IDXS["left_eye"], shape)
    mask = eye_on_mask(mask, FACIAL_LANDMARKS_IDXS["right_eye"], shape)
    mask = cv2.dilate(mask, kernel, 5) #膨胀处理
    eyes = cv2.bitwise_and(img, img, mask=mask) #图像每个像素二进制与
    mask = ((eyes == [0, 0, 0]).all(axis=2))
    eyes[mask] = [255, 255, 255]
    mid = (shape[42][0] + shape[39][0]) // 2
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2) #1
    thresh = cv2.dilate(thresh, None, iterations=4) #2
    thresh = cv2.medianBlur(thresh, 3) #3
    thresh = cv2.bitwise_not(thresh)
    center_dict = {}
    center_dict['right_eye_center'] = contouring(thresh[:, 0:mid], mid, img)
    center_dict['left_eye_center'] = contouring(thresh[:, mid:], mid, img, True)
    
    center_dict['right_eye_distance'] = None
    center_dict['left_eye_distance'] = None
    
    if center_dict['right_eye_center'] != None:
        center_dict['right_eye_distance'] = check_point_in_eye(center_dict['right_eye_center'], FACIAL_LANDMARKS_IDXS["right_eye"], shape)
    if center_dict['left_eye_center'] != None:
        center_dict['left_eye_distance'] = check_point_in_eye(center_dict['left_eye_center'], FACIAL_LANDMARKS_IDXS["left_eye"], shape)
    curr_center.append(center_dict)
    # face = np.matrix([[p.x, p.y] for p in face.parts()])
    curr_face_center.append(shape[27])

    out_list.append({'data':curr_center, 'ts':0, 'face_center': curr_face_center})
    return out_list

def get_eyeball(frame,keypoints):
    # eyeball_x,eyeball_y = -1,-1
    eyeball = eyeballs_track(frame,keypoints)
    curr_eyeball = eyeball[0]['data']
    curr_face_center = eyeball[0]['face_center']
    if len(curr_eyeball) == 0:
        eyeball_x = -1
        eyeball_y = -1
    elif (curr_eyeball[0]['right_eye_center'] != None and curr_eyeball[0]['left_eye_center'] != None):
        eyeball_x = int((curr_eyeball[0]['right_eye_center'][0] + curr_eyeball[0]['left_eye_center'][0]) / 2 - \
            curr_face_center[0][0])
        eyeball_x = abs(eyeball_x)
        eyeball_y = int((curr_eyeball[0]['right_eye_center'][1] + curr_eyeball[0]['left_eye_center'][1]) / 2 - \
            curr_face_center[0][1])
        eyeball_y = abs(eyeball_y) 
    else:
        eyeball_x = -1
        eyeball_y = -1
    return eyeball_x,eyeball_y
    