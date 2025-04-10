# -- coding: utf-8 --
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import sys
import msvcrt
import base64
import datetime
import logging

# sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *
from JsonResponse import *
from JsonFlask import *

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='hikrobot.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

# 这里配置一下 template_folder为当前目录，不然可以找不到 index.html
app = JsonFlask(__name__, template_folder='.')

# index
@app.route('/')
def index():
    return render_template('./templates/index.html')

# 获取码流
def generate(cap):
    """
    从摄像头抓取视频流并进行 YOLOv8 检测。
    """
    try:
        while True:
            # 如果关闭相机，退出循环
            global open
            if not open:
                break

            retgrab = cap.grab()
            if retgrab:
                logging.debug("Grab true")
            ret1, frame = cap.retrieve()
            if frame is None:
                logging.error("frame is None")
                continue

            # 使用 YOLOv8 对帧进行检测
            results = model.predict(frame, conf=0.5, show=False)  # 设置置信度阈值
            annotated_frame = results[0].plot()  # 绘制检测结果

            # 将检测结果编码为 JPEG 格式
            ret1, jpeg = cv2.imencode('.jpg', annotated_frame)
            jpg_frame = jpeg.tobytes()

            # 生成视频流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_frame + b'\r\n')
    except Exception as e:
        logging.error("generate error: %s" % str(e))

# 开始预览
@app.route('/startPreview')
def startPreview():
    logging.info("======================================")
    logging.info("start to preview video stream, current_time: " + str(datetime.datetime.now()))

    # 全局变量，用于控制获取视频流的开关状态
    global open
    open = True

    # 全局变量，获取视频连接
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("can't open camera")
        quit()
    else:
        logging.info("start to open camera")

    logging.info("open camera ok")

    # 分辨率设置 2448*2048（海康机器人工业相机 MV-CS050-10GC）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    cap.set(cv2.CAP_PROP_FPS, 20)

    # 初始化 YOLOv8 模型
    global yolo_model
    yolo_model = YOLO("yolov8n.pt")  # 替换为你的模型路径

    return Response(generate(cap), mimetype='multipart/x-mixed-replace;boundary=frame')

# 停止预览
@app.route('/stopPreview')
def stopPreview():
    logging.info("======================================")
    logging.info("stop to preview video stream, current_time: " + str(datetime.datetime.now()))
    logging.info("start to close camera")

    # 全局变量，用于停止循环
    global open
    open = False

    logging.info("release resources start")
    # 全局变量，用于释放相机资源
    try:
        global cap
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error("stopPreview error: %s" % str(e))
    logging.info("release resources end")
    logging.info("camera closed successfully, current_time: " + str(datetime.datetime.now()))
    logging.info("======================================")
    return JsonResponse.success()

# 获取一张图片 原理从opencv中读取一帧然后保存到本地
@app.route('/getImage')
def getImage():
    logging.info("======================================")
    logging.info("get Image start")
    _, frame = cap.read()  # 从 OpenCV 读取一帧

    # 使用 YOLOv8 对帧进行检测
    results = yolo_model.predict(frame, conf=0.5, show=False)
    annotated_frame = results[0].plot()  # 绘制检测结果

    # 保存检测结果到本地
    cv2.imwrite("test.jpg", annotated_frame)
    return JsonResponse.success()

# 执行web服务, 端口号可自行修订
logging.info("start to run camera app, current_time: " + str(datetime.datetime.now()))
# port 可以更改指定端口号
model = YOLO('../weights/yolov8l.pt')  # load a pretrained model (recommended for training)
app.run(host='0.0.0.0', port=65432, debug=True, threaded=True)
