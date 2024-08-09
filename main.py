import sys
import cv2 as cv
import time
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QDir, Qt

global widget
global cap
global msgBox

cap = None

class CustomQWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        self.vbox = QVBoxLayout(self)

def resizeWithAspectRatio(frame, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv.resize(frame, dim, interpolation=inter)

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, text, location, color=(50, 170, 50)):
    cv.putText(frame, text, location, cv.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def detectObjects(net, frame, dim = 300):
    blob = cv.dnn.blobFromImage(frame, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    
    return objects

def displayText(frame, text, x, y, FONTFACE, FONT_SCALE, THICKNESS):
    textSize = cv.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
    
    cv.rectangle(frame, 
                 (x, y - dim[1] - baseline),
                 (x + dim[0], y + baseline),
                 (0, 0, 0),
                 cv.FILLED,
    )
    
    cv.putText(frame,
               text,
               (x, y - 5),
               FONTFACE,
               FONT_SCALE,
               (0, 255, 255),
               THICKNESS,
               cv.LINE_AA
    )

def drawObjects(frame, objects, labels, FONTFACE, FONT_SCALE, THICKNESS, THRESHOLD):
    rows = frame.shape[0]
    cols = frame.shape[1]
    
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        
        if score > THRESHOLD:
            displayText(frame, "{}".format(labels[classId]), x, y, FONTFACE, FONT_SCALE, THICKNESS)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

def drawPose(frame, net, POSE_PAIRS, THRESHOLD):
    points = []
    nPoints = 15
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    
    netInputSize = (658, 368)
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)
    
    output = net.forward()
    
    scaleX = inWidth / output.shape[3]
    scaleY = inHeight / output.shape[2]
    
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        _, prob, _, point = cv.minMaxLoc(probMap)
        
        x = scaleX * point[0]
        y = scaleY * point[1]
        
        if prob > THRESHOLD:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        
        if points[partA] and points[partB]:
            cv.line(frame, points[partA], points[partB], (255, 255, 0), 2)
            cv.circle(frame, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv.FILLED)
            cv.circle(frame, points[partB], 8, (255, 0, 0), thickness=-1, lineType=cv.FILLED)

def window():
    global widget
    global msgBox
    
    winWidth = 480
    winHeight = 270
    
    app = QApplication(sys.argv)
    widget = CustomQWidget()
    widget.setStyleSheet("background-color: darkslategrey;"
                         "color: k;")
    widget.setFixedSize(winWidth, winHeight)
    widget.setWindowIcon(QIcon("icon.png"))
    widget.setWindowTitle("OCVI")
    
    msgBox = QMessageBox()
    msgBox.setStyleSheet("background-color: darkslategrey;"
                         "color: k;")

    button1 = QPushButton()
    button1.setStyleSheet("background-color: lightgrey")
    button1.setText("Load")
    button1.clicked.connect(button1_clicked)
    widget.vbox.addWidget(button1, alignment=Qt.AlignHCenter)
    
    button2 = QPushButton()
    button2.setStyleSheet("background-color: lightgrey")
    button2.setText("Play")
    button2.clicked.connect(button2_clicked)
    widget.vbox.addWidget(button2, alignment=Qt.AlignHCenter)
    
    button3 = QPushButton()
    button3.setStyleSheet("background-color: gold")
    button3.setText("Motion Detection")
    button3.clicked.connect(button3_clicked)
    widget.vbox.addWidget(button3, alignment=Qt.AlignHCenter)

    button4 = QPushButton()
    button4.setStyleSheet("background-color: gold")
    button4.setText("Object Detection")
    button4.clicked.connect(button4_clicked)
    widget.vbox.addWidget(button4, alignment=Qt.AlignHCenter)
    
    button5 = QPushButton()
    button5.setStyleSheet("background-color: gold")
    button5.setText("Pose Detection")
    button5.clicked.connect(button5_clicked)
    widget.vbox.addWidget(button5, alignment=Qt.AlignHCenter)

    widget.show()
    sys.exit(app.exec_())

def button1_clicked():
    global widget
    global cap
    global msgBox
    
    fileName = QFileDialog.getOpenFileName(widget, "Open Video", QDir.home().dirName(), "Video Files (*.mp4 *.mkv *.avi)")
    if fileName[0]:
        widget.setWindowTitle("OCVI - " + fileName[0].split("/")[-1])
        cap = cv.VideoCapture(fileName[0])

def button2_clicked():
    global cap
    global msgBox
    
    if cap is None:
        msgBox.setText("No Video Loaded")
        msgBox.exec()
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    TIMEOUT = 1/fps
    oldTimestamp = time.time()
    
    windowName = "Preview"
    while cap.isOpened():
        if (time.time() - oldTimestamp) > TIMEOUT:
            oldTimestamp = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            resFrame = resizeWithAspectRatio(frame, width=480, inter=cv.INTER_AREA)
        
            cv.imshow(windowName, resFrame)
            if cv.waitKey(1) > 0 or cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) < 1:
                break  
            
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    cv.destroyAllWindows()

def button3_clicked():
    global cap
    global widget
    global msgBox
    
    if cap is None:
        msgBox.setText("No Video Loaded")
        msgBox.exec()
        return
    
    tracker_types = [
        "CSRT",
        "MIL",
        "NANO",
    ]
    
    text, ok = QInputDialog.getText(widget, "Input", "Tracker Type [CSRT,MIL,NANO]:", QLineEdit.Normal)
    if ok:
        if text in tracker_types:
            tracker_type = text
        else:
            msgBox.setText("Incorrect Tracker Type")
            msgBox.exec()
            return
            
    else:
        return
    
    if tracker_type == "CSRT":
        tracker = cv.TrackerCSRT.create()
    elif tracker_type == "MIL":
        tracker = cv.TrackerMIL.create()
    elif tracker_type == "NANO":
        params = cv.TrackerNano_Params()
        params.backbone = os.path.join("models", "nanotrack_backbone_sim.onnx")
        params.neckhead = os.path.join("models", "nanotrack_head_sim.onnx")
        tracker = cv.TrackerNano.create(params)
    
    ret, frame = cap.read()
    if not ret:
        msgBox.setText("Could Not Open Video")
        msgBox.exec()
        return
   
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
   
    bbox = (int(width/4), int(height/4), int(width/4), int(width/4))
    tracker.init(frame, bbox)
    
    out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'mp4v'), cap.get(cv.CAP_PROP_FPS), (width, height))
    
    windowName = "Motion Detection"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        ok, bbox = tracker.update(frame)
        
        if ok:
            drawRectangle(frame, bbox)
        else:
            drawText(frame, "Tracking Failure", (80, 140), (0, 0, 255))
        
        drawText(frame, "Tracker : " + tracker_type, (80, 60))
        
        out.write(frame)
        
        resFrame = resizeWithAspectRatio(frame, width=480, inter=cv.INTER_AREA)
        cv.imshow(windowName, resFrame)
        if cv.waitKey(1) > 0 or cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) < 1:
            break
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    out.release()
    cv.destroyAllWindows()
    msgBox.setText("Result Saved To 'output.mp4'")
    msgBox.exec()

def button4_clicked():
    global cap
    global widget
    global msgBox
    
    if cap is None:
        msgBox.setText("No Video Loaded")
        msgBox.exec()
        return
    
    classFile = os.path.join("models", "coco_class_labels.txt")
    with open(classFile) as fp:
        labels = fp.read().split("\n")
    
    modelFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
    configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
    
    FONTFACE = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1
    
    num, ok = QInputDialog.getDouble(widget, "Input", "Threshold:", 0.1, 0.1, 1.0, 2, Qt.WindowFlags(), 0.05)
    
    if ok:
        THRESHOLD = num
        
    else:
        return
    
    out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'mp4v'), cap.get(cv.CAP_PROP_FPS), (width, height))
    
    windowName = "Object Detection"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        objects = detectObjects(net, frame)
        drawObjects(frame, objects, labels, FONTFACE, FONT_SCALE, THICKNESS, THRESHOLD)
        
        drawText(frame, "Threshold : " + str(THRESHOLD), (80, 60))
        
        out.write(frame)
        
        resFrame = resizeWithAspectRatio(frame, width=480, inter=cv.INTER_AREA)
        cv.imshow(windowName, resFrame)
        if cv.waitKey(1) > 0 or cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) < 1:
            break
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    out.release()
    cv.destroyAllWindows()
    msgBox.setText("Result Saved To 'output.mp4'")
    msgBox.exec()

def button5_clicked():
    global cap
    global widget
    global msgBox
    
    if cap is None:
        msgBox.setText("No Video Loaded")
        msgBox.exec()
        return
    
    protoFile = os.path.join("models", "pose_deploy.prototxt")
    weightsFile = os.path.join("models", "pose_iter_584000.caffemodel")
    
    POSE_PAIRS = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 5],
        [5, 6],
        [6, 7],
        [1, 14],
        [14, 8],
        [8, 9],
        [9, 10],
        [14, 11],
        [11, 12],
        [12,13],
    ]
    
    num, ok = QInputDialog.getDouble(widget, "Input", "Threshold:", 0.1, 0.1, 1.0, 2, Qt.WindowFlags(), 0.05)
    
    if ok:
        THRESHOLD = num
        
    else:
        return
    
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    out = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'mp4v'), cap.get(cv.CAP_PROP_FPS), (width, height))
    
    windowName = "Pose Detection"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        drawPose(frame, net, POSE_PAIRS, THRESHOLD)
        
        drawText(frame, "Threshold : " + str(THRESHOLD), (80, 60))
        
        out.write(frame)
        
        resFrame = resizeWithAspectRatio(frame, width=480, inter=cv.INTER_AREA)
        cv.imshow(windowName, resFrame)
        if cv.waitKey(1) > 0 or cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) < 1:
            break
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    out.release()
    cv.destroyAllWindows()
    msgBox.setText("Result Saved To 'output.mp4'")
    msgBox.exec()

if __name__ == '__main__':
    window()
    cap.release()
    cv.destroyAllWindows()
