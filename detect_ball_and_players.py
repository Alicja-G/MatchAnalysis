import torch
import cv2

# Path to your YOLOv5 model weights
MODEL_WEIGHTS = 'yolov5s.pt'

# Classes of interest: person=0, sports ball=32 in COCO
PLAYER_CLASS_ID = 0
BALL_CLASS_ID = 32

# Load YOLOv5 model
# This uses the Ultralytics hub. If you prefer, you can also do:
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS)
model.conf = 0.05  # Confidence threshold, adjust as needed

# Path to input video
input_video = 'D:\mecze MJO dron\cutVersion.mp4'  # Replace with your actual video path
output_video = 'results_detected3.mp4'

# OpenCV video capture
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference with YOLOv5
    results = model(frame, size=640)  # You can adjust size for speed/accuracy
    detections = results.xyxy[0].numpy()  # [x1, y1, x2, y2, conf, cls]

    # Filter detections for players and balls
    player_detections = [d for d in detections if int(d[5]) == PLAYER_CLASS_ID]
    ball_detections = [d for d in detections if int(d[5]) == BALL_CLASS_ID]

    # Draw rectangles and labels for players
    for det in player_detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, "Player", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Draw rectangles and labels for ball
    for det in ball_detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(frame, "Ball", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Write the frame to output
    out.write(frame)

    # Show the video in a window (optional)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
