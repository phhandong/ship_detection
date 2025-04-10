import cv2
import time
import multiprocessing as mp

"""
Source: Yonv1943 2018-06-17
https://github.com/Yonv1943/Python/tree/master/Demo
"""

def image_put(q, name, pwd, ip, channel=1):
    # 初始化 RTSP 流
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    if cap.isOpened():
        print(f'HIKVISION Camera connected: {ip}')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
        print(f'DaHua Camera connected: {ip}')

    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame)  # 将帧放入队列
        q.get() if q.qsize() > 1 else time.sleep(0.01)  # 丢弃旧帧，保持队列最新

def image_get(q, window_name):
    # 显示视频流
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        if not q.empty():
            frame = q.get()  # 从队列中获取最新帧
            if frame is not None:
                # 调整分辨率（可选）
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
    cv2.destroyAllWindows()

def run_multi_camera():
    # 用户名和密码
    user_name, user_pwd = "admin", "Hczk406."
    # 摄像头 IP 地址列表
    camera_ip_l = [
        "192.168.2.200",  # 替换为你的摄像头 IP 地址
    ]

    mp.set_start_method(method='spawn')  # 初始化多进程
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]  # 为每个摄像头创建队列

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        # 创建图像抓取进程
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        # 创建图像显示进程
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()

if __name__ == '__main__':
    run_multi_camera()