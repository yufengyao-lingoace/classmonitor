import ffmpeg
import time
import os
import cv2
import imageio
from numpy.core.arrayprint import _void_scalar_repr
from OneImageAna import OneImageAna
import clothing
import human_position
from hand_utils import get_hand_ts
from pydub import AudioSegment
import numpy as np
import array
import math
import json
import threading
import requests
import bucketstore
from wav_utils import ASR
from flask import Flask, request

app = Flask(__name__)
bucketstore.login('AKIASMGRD4KPGKSBHDBA', 'zCzCUVm0VqPfpAXTaURCDGGA2eyB3+cvyFyM9lCX', 'cn-north-1')
current_bucket = bucketstore.get('media.ppchinese.com')
REMOTE_FOLDER = 'lingchuntest/'

asr=ASR() #语音识别引擎
contours, con_y_max, con_y_min = human_position.process_frame_figure('position.png')

def download_video(in_file, out_mp4, out_wav):
    try:
        if os.path.exists(out_mp4):
            os.remove(out_mp4)
        video = ffmpeg.input(in_file)
        video_mp4 = ffmpeg.output(
            video, out_mp4, absf='aac_adtstoasc', f='mp4', acodec='copy', vcodec='copy')
        video_mp4.run()
        return True
        # audio = video.audio
        # audio_output = ffmpeg.output(audio, wav_output_source, ac=1, ar=8000)
        # audio_output.run()
    except Exception as ex:
        print("download error:"+str(ex))
        return False

def sendData(callback,data,fid):
    result=False
    try:
        print('开始上传分析结果[{0}]...'.format(fid))
        data = json.dumps(data)  
        headers={'Content-Type':'application/json; charset=UTF-8'}
        for i in range(5):
            print('尝试第{0}次上传...'.format(i+1))
            res = requests.post(url=callback,data=data,headers=headers,timeout=30) #上传结果
            code=res.status_code
            if code==200: #post成功
                result=True
                print('上传成功...')  
                break
            print('第{0}次上传结果：{1}'.format(i+1,str(res)))
    except Exception as ex:
        print('上传分析结果失败!错误信息：{0}'.format(ex))
        result=False
    finally:
        return result

def upload_video(out_file_name, video_file, file_path, remove_local = True):
    remote_path = REMOTE_FOLDER + out_file_name
    video_content = current_bucket.key(remote_path) # create key
    with open(os.path.join(file_path, video_file), 'rb') as f: # upload file
        video_content.upload(f)
    video_content.make_public()
    if remove_local:
        os.remove(os.path.join(file_path, video_file))
    return video_content.url

def video_ana(curr_args):
        job_id = curr_args['jobId']
        room_id = curr_args['roomId']
        user_id = curr_args['data']['userId']
        url = curr_args['data']['recordingUrl']
        class_s = curr_args['data']['startTime']
        class_e = curr_args['data']['endTime']
        FINISH_CALL_URL_MOMENT = '/eduadmin/api/ai/callback/highlight'
        upload_url=curr_args['callbackDomain'] + FINISH_CALL_URL_MOMENT #上传分析结果url

        result={"jobId": job_id,"roomId": room_id,"userId":user_id,
                "tea_detail":[],"stu_detail":[],"highlights":[],
                "tea_behavior":{"closeTs": 0, "farTs": 0, "verticalTs": 0, "horiTs": 0, "necklineTs": [], "gestureTs": []},
                "tea_img_url":"","stu_img_url":""}
        #tea_detail:["facecount","emo","attention","mouse","eye",x,y]]
        ret = download_video(url, "tmp.mp4", "")  # 下载mp4文件
        if not ret:  # 下载失败
            return "download video failed!"

    # try:
        detector = OneImageAna()
        file = "tmp.mp4"
        videoCapture = cv2.VideoCapture(file)  # 打开视频
        fps = int(videoCapture.get(cv2.CAP_PROP_FPS))  # 视频码率
        v_size = [int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)]  # 视频尺寸
        img_size = v_size[1]
        scaled_size = 240 #缩小后的尺寸
        border_size = 50 #加边
        detect_size=scaled_size+2*border_size #缩放后+加边
        fNUMS = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
        videoCapture.release()
        try:
            audio = AudioSegment.from_file(file, "mp4")
        except:
            print("open audio failed!")
            audio = None
        try:
            reader = imageio.get_reader(file, 'ffmpeg')
        except:
            print("open video failed!")
            return  # 过滤无效视频
        
        max_emo_score_stu, max_emo_score_tea, max_emo_frame_stu, max_emo_frame_tea = 0, 0, None, None
        
        emotion_list,attention_tea_tmp,attention_stu_tmp=[],[],[]
        for index, frame_rgb in enumerate(reader):
            if index % fps != 0:
                continue
            mouse_tea,mouse_tea_dis,mouse_stu,mouse_stu_dis,eye_tea,eye_stu,emo_score_tea,emo_score_stu=-1,-1,-1,-1,-1,-1,-1,-1 #初始化值
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_stu = frame[img_size:, :, :]
            frame_tea = frame[0:img_size, :, :]

            frame_stu = cv2.resize(frame_stu, (scaled_size, scaled_size),interpolation=cv2.INTER_LINEAR)
            frame_tea = cv2.resize(frame_tea, (scaled_size, scaled_size),interpolation=cv2.INTER_LINEAR)
            frame_stu = cv2.copyMakeBorder(frame_stu, border_size, border_size, border_size,border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像边缘扩展
            frame_tea = cv2.copyMakeBorder(frame_tea, border_size, border_size, border_size,border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])  # 图像边缘扩展

            r_stu, face_count_stu = detector.getResult(frame_stu, border=0)
            r_tea, face_count_tea = detector.getResult(frame_tea, border=0)
            
            ##############################################################表情提取
            if len(r_stu) > 0:
                eye_stu,mouse_stu,mouse_stu_dis=r_stu[0]["eye"][0],r_stu[0]["mouse"][0],r_stu[0]["mouse"][1]
                emo = r_stu[0]["emo"]
                # emotion_stu["data"].append({"angry": emo[1][0], "fear": emo[1][2], "happy": emo[1][3], "neutral": emo[1][6], "sad": emo[1][4], "surprise": emo[1][5], "time": len(emotion_stu["data"])})
                emo_score_stu = round((0.059785*emo[1][0]+0.027026*emo[1][1]+0.039953*emo[1][2]+0.368842*emo[1][3]+0.059785 * emo[1][4]+0.260724*emo[1][5]+0.183885*emo[1][6])/0.368842,2)
                if emo_score_stu > max_emo_score_stu:
                    max_emo_score_stu = emo_score_stu
                    max_emo_frame_stu = frame
            if len(r_tea) > 0:
                eye_tea,mouse_tea,mouse_tea_dis=r_tea[0]["eye"][0],r_tea[0]["mouse"][0],r_tea[0]["mouse"][1]
                emo = r_tea[0]["emo"]
                if len(emo) > 0:
                    # emotion_tea["data"].append({"angry": emo[1][0], "fear": emo[1][2], "happy": emo[1][3], "neutral": emo[1][6], "sad": emo[1][4], "surprise": emo[1][5], "time": len(emotion_tea["data"])})
                    emo_score_tea = round((0.059785*emo[1][0]+0.027026*emo[1][1]+0.039953*emo[1][2]+0.368842*emo[1][3]+0.059785 * emo[1][4]+0.260724*emo[1][5]+0.183885*emo[1][6])/0.368842,2)
                    if emo_score_tea > max_emo_score_tea:
                        max_emo_score_tea = emo_score_tea
                        max_emo_frame_tea = frame
            if len(r_stu) > 0 and len(r_tea) > 0: #精彩瞬间分数
                emotion_list.append(0.4*emo_score_tea+0.6*emo_score_stu)
            else:
                emotion_list.append(np.nan)
            ##############################################################视线专注度提取
            attention_score_tea,attention_score_stu=0,0
            x_tea,y_tea,x_stu,y_stu=-1,-1,-1,-1
            if len(r_tea) > 0:
                p=r_tea[0]["nose_end"]
                attention_tea_tmp.append([p[0],p[1]])
                x_tea,y_tea=round(p[0]/detect_size,2),round(p[1]/detect_size,2)
            else:
                attention_tea_tmp.append([np.nan,np.nan])

            if len(r_stu) > 0:
                p=r_stu[0]["nose_end"]
                attention_stu_tmp.append([p[0],p[1]])
                x_stu,y_stu=round(p[0]/detect_size,2),round(p[1]/detect_size,2)
            else:
                attention_stu_tmp.append([np.nan,np.nan])
            if index>30: #计算专注度得分
                attention_score_tea=cal_attention_score(attention_tea_tmp)
                attention_score_stu=cal_attention_score(attention_stu_tmp)

            result["tea_detail"].append([face_count_tea,emo_score_tea,attention_score_tea,mouse_tea,mouse_tea_dis,eye_tea,x_tea,y_tea])
            result["stu_detail"].append([face_count_stu,emo_score_stu,attention_score_stu,mouse_stu,mouse_stu_dis,eye_stu,x_stu,y_stu])
            # 面部特征提取
            # eyeball = eyeballs_track(face_features)
            # if len(r_stu) > 0:
            #     face_stu["data"].append({"eyeAspectRatio": r_stu[0]["eye"][0], "eyeX": r_stu[0]["eye"][1], "eyeY": r_stu[0]["eye"][2],
            #                             "monthAspectRatio": r_stu[0]["mouse"][1], "faceCount": face_num_stu, "time": len(face_stu["data"])})
            # if len(r_tea) > 0:
            #     face_tea["data"].append({"eyeAspectRatio": r_tea[0]["eye"][0], "eyeX": r_tea[0]["eye"][1], "eyeY": r_tea[0]["eye"][2],
            #                             "monthAspectRatio": r_tea[0]["mouse"][1], "faceCount": face_num_tea, "time": len(face_tea["data"])})
            ##############################################################teacher行为提取
            if len(r_tea) > 0:
                problems = human_position.positon_check(r_tea[0]["keypoints"], contours, con_y_max, con_y_min)
                if problems[0] == 1: #距离过近
                    # result["tea_behavior"]["closeTs"].append(index//15)
                    result["tea_behavior"]["closeTs"]+=1
                if problems[1] == 1: #人像水平偏移
                    # result["tea_behavior"]["horiTs"].append(index//15)
                    result["tea_behavior"]["horiTs"]+=1
                if problems[2] == 1: #距离过远
                    # result["tea_behavior"]["farTs"].append(index//15)
                    result["tea_behavior"]["farTs"]+=1
                if problems[3] == 1: #人像竖直偏移
                    # result["tea_behavior"]["verticalTs"].append(index//15)
                    result["tea_behavior"]["verticalTs"]+=1
                cloth_ts = clothing.clothingExposureLevel(frame_tea, r_tea[0]["keypoints"], 15, 1.8)
                if cloth_ts != -1: #领口问题
                    result["tea_behavior"]["necklineTs"].append(index//15)
            hand_ts = get_hand_ts(frame_tea)
            if hand_ts is not None:
                    result["tea_behavior"]["gestureTs"].append([index//15,hand_ts])
            # if hand_ts: #包含手势
            #     result["tea_behavior"]["gestureTs"].append(index//15)
        cv2.imwrite("tmp/"+job_id+'_student.jpg', max_emo_frame_stu[360:, :, :])
        cv2.imwrite("tmp/"+job_id+'_teacher.jpg', max_emo_frame_tea[:360, :, :])
        ##############################################################精彩瞬间提取
        emo_score_list = []
        audio_score_list = []
        index_list = []
        for i in range(0, len(emotion_list)-20, 5):
            index_list.append(i)
            start_time = 1000*i  # 毫秒
            end_time = start_time+15000
            audio_period = audio[start_time:end_time]
            data = np.abs(array.array(audio_period.array_type, audio_period._data))
            audio_score_list.append(np.sum(data > 500))
        audio_score_list = (audio_score_list - np.nanmin(audio_score_list)) / (np.nanmax(audio_score_list) - np.nanmin(audio_score_list))  # normalization
        for i in range(0, len(emotion_list)-20, 5):
            emo_score_list.append(np.nanmean(emotion_list[i:i+15]))
        emo_score_list = (emo_score_list - np.nanmin(emo_score_list)) / (np.nanmax(emo_score_list) - np.nanmin(emo_score_list))  # normalization
        total_score_list = 0.7*emo_score_list+0.3*audio_score_list

        total_score_list = list(zip(index_list, total_score_list))
        total_score_list = [item for item in total_score_list if not math.isnan(item[1])]
        sorted_list = sorted(total_score_list, key=lambda x: x[1], reverse=True)

        happy_list = []
        history_time_list = [0]
        for item in sorted_list:
            cur_time = item[0]
            if np.sum(np.abs(np.array(history_time_list)-cur_time) < 15) > 0:
                continue
            else:
                happy_list.append(item)
                history_time_list.append(item[0])
        highlight_data=[] #精彩瞬间数据
        for i, item in enumerate(happy_list[:10]):
            tmp_mp4_name = job_id + '-' + str(int(item[0])) + '-' + str(int(item[0]+15)) + '-' + str(i) + '.mp4'
            ffmpeg_cut(file, "tmp/"+tmp_mp4_name, (int(item[0]), int(item[0]+15)))
            url_str = upload_video(job_id + '-' + str(int(item[0])) + '-' + str(int(item[0]+15)) +'.mp4', tmp_mp4_name, "tmp")
            highlight_data.append({'videoUrl':url_str, 'startTime':int(item[0]), 'endTime':int(item[0]+15), 'score':round(float(item[1]), 2)})
        ##############################################################语音提取,区分老师和学生语音
        wav=asr.recognize_wav(audio)
        wav_stu,wav_tea=[],[]
        for item in wav:
            t1,t2=item[0],item[1]
            person=get_wav_person(t1,t2,result)
            if person==1: #tea
                wav_tea.append(item)
            else:
                wav_stu.append(item)
            print(str(person)+"    "+item[2])
            
        result["wav_tea"]=wav_tea
        result["wav_stu"]=wav_stu
        #####上传数据
        # sendData(curr_args['callbackDomain'] + FINISH_CALL_URL_EMOTION,emotion_stu,job_id)
        # sendData(curr_args['callbackDomain'] + FINISH_CALL_URL_EMOTION,emotion_tea,job_id)
        # sendData(curr_args['callbackDomain'] + FINISH_CALL_URL_FACE,emotion_stu,job_id)
        # sendData(curr_args['callbackDomain'] + FINISH_CALL_URL_FACE,emotion_tea,job_id)
        # sendData(curr_args['callbackDomain'] + FINISH_CALL_URL_TEACHER,result.tea_behavior,job_id)
        teacher_url = upload_video(job_id+'_teacher.jpg', job_id+'_teacher.jpg', "tmp")
        student_url = upload_video(job_id+'_student.jpg', job_id+'_student.jpg', "tmp")
        result["highlights"]=highlight_data
        result["tea_img_url"]=teacher_url
        result["stu_img_url"]=student_url
        sendData(upload_url,result,job_id) #上传分析结果
        with open("result.txt",'w') as f:
            f.write(json.dumps(result).replace(" ",""))
        
        print("分析完成...")

    # except Exception as ex:
    #     print(ex)
    # finally:
    #     pass
        # return emotion_stu, emotion_tea, face_stu, face_tea, tea_behavior

def is_history(cur_time, history_time_list):
    for item in history_time_list:
        if abs(cur_time-item) <= 15:
            return True
    return False

def cal_attention_score(attention_list):
    try:
        mean_point=np.nanmean(attention_list[-20:],axis=0)
        center_point=list(mean_point)
        if center_point[0]==np.nan or center_point[1]==np.nan:
            center_point=[170,170]
        score=0
        num=0 #有效个数
        for i in range(len(attention_list)):
            p1=attention_list[-1-i]
            if p1[0]==np.nan:
                continue
            dis=math.sqrt((p1[0]-center_point[0])**2+(p1[1]-center_point[1])**2)
            score_tmp=1-dis/150
            if score_tmp<0:
                score_tmp=0
            score+=score_tmp 
            num+=1
            if num>=5:
                break
        score=round(score/num,2)
        score=score if score>0 else 0
        return score
    except Exception as ex:
        print("cal_attention_score:"+str(ex))
        return 0

def get_wav_person(t1,t2,data):
    t1=t1//1000
    t2=t2//1000
    if t1==t2:
        t2=t2+1
    
    tea_mouse=np.array(data["tea_detail"][t1:t2])
    tea_mouse=tea_mouse[:,3]
    stu_mouse=np.array(data["stu_detail"][t1:t2])
    stu_mouse=stu_mouse[:,3]
    # r=1 if np.var(tea_mouse)>=np.var(stu_mouse) else 2
    r=1 if np.mean(tea_mouse)>=np.mean(stu_mouse) else 2
    return r

def ffmpeg_cut(input_source, output_source, time_window):
    start_sec, end_sec = time_window
    input_vid = ffmpeg.input(input_source)
    vid = (
        input_vid
        .trim(start=start_sec, end=end_sec)
        .setpts('PTS-STARTPTS')
    )
    aud = (
        input_vid
        .filter_('atrim', start=start_sec, end=end_sec)
        .filter_('asetpts', 'PTS-STARTPTS')
    )
    joined = ffmpeg.concat(vid, aud, v=1, a=1).node
    output = ffmpeg.output(joined[0], joined[1], output_source)
    output.run()

@app.route('/ai/casia/moment', methods=['POST'])
def moment_generate():
    '''
    roomId
    userId
    recordingUrl
    startTime
    endTime
    '''
    curr_args = json.loads(request.get_data(as_text=True))
    try:
        name_id = curr_args['jobId']
        room_id = curr_args['roomId']
        user_id = curr_args['data']['userId']
        video_path = curr_args['data']['recordingUrl']
        class_s = curr_args['data']['startTime']
        class_e = curr_args['data']['endTime']
        thread_process = threading.Thread(target=video_ana, args=(curr_args,))
        thread_process.start()
    except:
        return "argument wrong"

    return "DONE"

def run():
    data = {
        "jobId": '2',
        "roomId": '2',
        "userId": '2',
        "data": {
            "userId": "1",
            # "recordingUrl": "https://s3-us-west-1.amazonaws.com/media.pplingo.com/lingorecordings/10073/273941cbdb4ec5043766578cd9531667_10073.m3u8",
            "recordingUrl": "https://s3.cn-north-1.amazonaws.com.cn/media.ppchinese.com/lingchuntest/499/83f678620747c1c8c1553cb5f6f20801_499.m3u8",
            "startTime": "1",
            "endTime": "1",
        },
        "callbackDomain": "https://test234-classroom.lingo-ace.com"
    }
    # current_m3u8 = "https://s3-us-west-1.amazonaws.com/media.pplingo.com/lingorecordings/698/5de8c7ae4747c6d1893a188b0a451a92_698.m3u8" #6分钟
    # current_m3u8 = "https://s3.cn-north-1.amazonaws.com.cn/media.ppchinese.com/lingchuntest/499/83f678620747c1c8c1553cb5f6f20801_499.m3u8"
    # current_m3u8 = "https://s3-us-west-1.amazonaws.com/media.pplingo.com/lingorecordings/1430/f3345d244a428e713136a3b62a62febe_1430.m3u8"

    t3 = time.time()
    video_ana(data)
    t4 = time.time()
    print('分析时间：{0}秒'.format(t4-t3))

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    run()
