import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np
from pathlib import Path
from test_gpu import test_cuda


BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "best.pt" #указываем версию YOLO, которую будем использовать

model = YOLO(model_path)
if test_cuda():
    model.to("cuda")

#def RGB(event, x, y):
#    if event == cv2.EVENT_MOUSEMOVE:
#        point = [x, y]
#        print(point)
  
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(BASE_DIR / "detect" / 'test_eclair_video.mp4')

my_file = open(BASE_DIR / "detect" / "find_object.txt", "r")
data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))

    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]

        if 'eclair' in c:
            list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cv2.circle(frame, (x4, y4), 4, (100, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)

    out.write(frame)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

