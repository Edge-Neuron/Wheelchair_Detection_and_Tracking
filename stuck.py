import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import gpiozero

buzzer = gpiozero.Buzzer(4)

model = YOLO(' path for wheel chair model ')

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1600, 1600), "format": "RGB888"})
picam2.configure(config)
picam2.start()

frame_width = 640
frame_height = 640
stationary_frames_threshold = 5
pixel_threshold = 30
index = 2  

prev_time = 0
buzzer_time = 0
centroid_history = []

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (frame_width, frame_height))

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    results = model(frame, imgsz=640, classes=[index])

    current_centroids = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.6:  
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                current_centroids.append((cX, cY))
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    centroid_history.append(current_centroids)
    if len(centroid_history) > stationary_frames_threshold:
        centroid_history.pop(0)

    if len(centroid_history) == stationary_frames_threshold:
        for i in range(len(current_centroids)):
            stationary = True
            for j in range(1, stationary_frames_threshold):
                try:
                    prev_cX, prev_cY = centroid_history[j - 1][i]
                    curr_cX, curr_cY = centroid_history[j][i]
                except IndexError:
                    stationary = False
                    break
                if abs(curr_cX - prev_cX) > pixel_threshold or abs(curr_cY - prev_cY) > pixel_threshold:
                    stationary = False
                    break
            if stationary and time.time() > buzzer_time:
                print('ys')
                buzzer.on()
                buzzer_time = time.time() + 2
                break
        else:
            buzzer.off()

    cv2.putText(frame, f"FPS: {fps:.2f}", (frame_width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

buzzer.off()
cv2.destroyAllWindows()
