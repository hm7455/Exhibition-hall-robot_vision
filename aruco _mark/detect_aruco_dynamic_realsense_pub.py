import numpy as np
import time
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import socket
import struct

# 相机参数
dist = np.array(([[0.0144544845, 1.30058745, -0.000653923289, -0.000750959340, -5.33048171]]))
newcameramtx = np.array([[189.076828, 0., 361.20126638],
                       [0, 2.01627296e+04, 4.52759577e+02],
                       [0, 0, 1]])
mtx = np.array([[609.21580774, 0., 326.90021511],
              [0., 609.0366553, 229.83638659],
              [0., 0., 1.]])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

# 与C++通信
#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client.connect(('192.168.0.71', 7730))


# 均值滤波
def mean_filter(pos_x_list,pos_y_list,pos_z_list):
    num_pos = len(pos_x_list)
    pos_x_num = 0
    pos_y_num = 0
    pos_z_num = 0
    for i in range(num_pos):
        pos_x_num = pos_x_num + pos_x_list[i]
        pos_y_num = pos_y_num + pos_y_list[i]
        pos_z_num = pos_z_num + pos_z_list[i]
    pos_x_mean = pos_x_num/num_pos
    pos_y_mean = pos_y_num / num_pos
    pos_z_mean = pos_z_num / num_pos
    return pos_x_mean,pos_y_mean,pos_z_mean

if __name__ == '__main__':
    number = 0
    while True:
        # 读取摄像头画面
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        h1, w1 = color_image.shape[:2]
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)     # aruco标定板类型
        parameters =  aruco.DetectorParameters_create()

        #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        #  如果相机视野中发现标定块
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.147, mtx, dist)    # 0.147为标定板的尺寸
            (rvec-tvec).any() # get rid of that nasty numpy value array error
            for i in range(rvec.shape[0]):
                aruco.drawAxis(color_image, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                aruco.drawDetectedMarkers(color_image, corners)
            cv2.putText(color_image, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

            if number == 0:
                pos_x = tvec[0][0][0]
                pos_y = tvec[0][0][0]
                pos_z = tvec[0][0][0]
                pos_x_list = np.array(pos_x)
                pos_y_list = np.array(pos_y)
                pos_z_list = np.array(pos_z)
                pos_x_mean = pos_x
                pos_y_mean = pos_y
                pos_z_mean = pos_z
            else:
                if number > 5:
                    pos_x_list = pos_x_list[1:4]
                    pos_x_list = np.append(pos_x_list,tvec[0][0][0])
                    pos_y_list = pos_y_list[1:4]
                    pos_y_list = np.append(pos_y_list, tvec[0][0][1])
                    pos_z_list = pos_z_list[1:4]
                    pos_z_list = np.append(pos_z_list, tvec[0][0][2])
                else:
                    pos_x_list = np.append(pos_x_list, tvec[0][0][0])
                    pos_y_list = np.append(pos_y_list, tvec[0][0][1])
                    pos_z_list = np.append(pos_y_list, tvec[0][0][2])
                pos_x_mean, pos_y_mean, pos_z_mean = mean_filter(pos_x_list,pos_y_list,pos_z_list)

            number = number+1
            # 发送数据
            data1 = 118
            data2 = 100   # 检测到标定块时置信度为100
            data0 = [0]*3
            data0[0] = pos_z_mean     # 相机坐标系转化到小车坐标系
            data0[1] = -pos_x_mean
            data0[2] = -pos_y_mean
            print("x:" + str(data0[0])+"y:" + str(data0[1])+"z:" + str(data0[2]))
        else:
            cv2.putText(color_image, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            data1 = 118
            data2 = 0   # 检测不到标定块时置信度为0
            data0 = [0] * 3
            data0[0] = 0
            data0[1] = 0
            data0[2] = 0
        # 打包数据，发送到服务端
        buf2 = struct.pack("iiddd", data1, data2, data0[0], data0[1], data0[2])
        #client.send(buf2)
        # 显示结果
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key == 27:         # 按esc键退出
            print('esc break...')
            cv2.destroyAllWindows()
            #client.close()    # 通信终止
            break
        if key == ord(' '):   # 按空格键保存
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, color_image)
