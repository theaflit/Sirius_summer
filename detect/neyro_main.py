import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np
from pathlib import Path
from test_gpu import check_device


def launch_yolo(model_name: str, 
                device: str, 
                video: bool, 
                video_name: str | None = None):

    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "models" / model_name # указываем версию YOLO, которую будем использовать

    model = YOLO(model_path)
    device = check_device(device)
    model.to(device)

    #def RGB(event, x, y):
    #    if event == cv2.EVENT_MOUSEMOVE:
    #        point = [x, y]
    #        print(point)
    
    # cv2.namedWindow('RGB')
    # cv2.setMouseCallback('RGB', RGB)

    source = BASE_DIR / "detect" / video_name if video else 0
    cap = cv2.VideoCapture(source)

    my_file = open(BASE_DIR / "detect" / "find_object.txt", "r")
    data = my_file.read()
    class_detections = data.split("\n")

    tracker = Tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame, verbose=False)
        a = results[0].boxes.data
        px = pd.DataFrame(a.cpu().numpy()).astype("float")

        detections = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            c = class_detections[d]

            if 'eclair' in c:
                detections.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(detections)

        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            cv2.circle(frame, (x4, y4), 4, (100, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    launch_yolo("best.pt", 'cuda', True, 'test_eclair_video.mp4')