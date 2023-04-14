# 匯入必要套件
import argparse
import time
from os.path import exists
from urllib.request import urlretrieve
import tensorflow as tf
import cv2
import numpy as np
from imutils.video import WebcamVideoStream

prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"


# 下載模型相關檔案
if not exists(prototxt) or not exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}",
                prototxt)
    urlretrieve(
        f"https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/{caffemodel}",
        caffemodel)
# 初始化模型 (模型使用的Input Size為 (96, 96))
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)

size = 96
model2 = tf.keras.models.load_model('mask_model_MobileNetV2(96_96).h5', compile=False)  # 載入模型(96*96)
data = np.ndarray(shape=(1, size, size, 3), dtype=np.float32)           #模型需要的格式

mask_label = {0:'MASK INCORRECT',1:'MASK', 2:'NO MASK'}
color = {0:(255,0,0),1:(0, 255,0), 2:(0,0,255)}                     #BGR

# 定義人臉偵測函數方便重複使用
def detect(img, min_confidence=0.5):
    # 取得img的大小(高，寬)
    (h, w) = img.shape[:2]

    # 建立模型使用的Input資料blob (比例變更為300 x 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 設定Input資料與取得模型預測結果
    net.setInput(blob)
    detectors = net.forward()

    # 初始化結果
    rects = []
    # loop所有預測結果
    for i in range(0, detectors.shape[2]):
        # 取得預測準確度
        confidence = detectors[0, 0, i, 2]

        # 篩選準確度低於argument設定的值
        if confidence < min_confidence:
            continue

        # 計算bounding box(邊界框)與準確率 - 取得(左上X，左上Y，右下X，右下Y)的值 (記得轉換回原始image的大小)
        box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
        # 將邊界框轉成正整數，方便畫圖
        (x0, y0, x1, y1) = box.astype("int")
        rects.append({"box": (x0, y0, x1 - x0, y1 - y0), "confidence": confidence})

    return rects



# 初始化arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter detecteions")
args = vars(ap.parse_args())

# 啟動WebCam
vs = WebcamVideoStream().start()
time.sleep(2.0)
start = time.time()
fps = vs.stream.get(cv2.CAP_PROP_FPS)
print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))

while True:
    # 取得當前的frame
    frame = vs.read()
    frame = cv2.resize(frame,(800,600))
    rects = detect(frame, args["confidence"])

    # loop所有預測結果
    for rect in rects:
        (x, y, w, h) = rect["box"]
        confidence = rect["confidence"]
        
        dx = int(x-w*0.2)        #擴大臉的範圍
        if dx<0:
            dx = 0
        dy = int(y-h*0.15)
        if dy<0:
            dy = 0
        dw = int(w*1.4)
        if dx+dw>800:
            dw = 800-dx
        dh = int(h*1.3)
        if dy+dh>600:
            dh = 600-dy
        
        imggg = frame[dx:dx+dw, dy:dy+dh]
        if not len(imggg) == 0:
            imggg = cv2.resize(imggg,(size,size))
            imggg = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)
            data[0] = np.asarray(imggg)
            data = data/255
            prediction = model2.predict(data)
            result = prediction[0].argmax()

        # 畫出邊界框
        
        if prediction[0][result]>0.5:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color[result], 2)   # 標記人臉
                cv2.putText(frame,mask_label[result]+'    {0:.2f}%'.format(prediction[0][result]*100),(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color[result],2)


    # 標示FPS
    end = time.time()
    cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    start = end

    # 顯示影像
    cv2.imshow("Frame", frame)

    # 判斷是否案下"q"；跳離迴圈
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break


