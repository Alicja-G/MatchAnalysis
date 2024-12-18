import torch
import cv2
import numpy as np
import sys
import os

DEEP_SORT_ROOT = os.path.join(os.getcwd(), 'deep_sort_pytorch')
if DEEP_SORT_ROOT not in sys.path:
    sys.path.append(DEEP_SORT_ROOT)

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# Classes of interest
PLAYER_CLASS_ID = 0      # "person"
BALL_CLASS_ID = 32       # "sports ball"
MODEL_WEIGHTS = 'yolov5s.pt'

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS)
model.conf = 0.1

# Setup Deep SORT
cfg = get_config()
cfg.merge_from_file(os.path.join(DEEP_SORT_ROOT, "configs/deep_sort.yaml"))

ckpt_path = r"D:\DRONE ANALYSIS\venv\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7"
deepsort = DeepSort(
    ckpt_path,
    max_dist=cfg.DEEPSORT.MAX_DIST, 
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE, 
    n_init=cfg.DEEPSORT.N_INIT, 
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=False,
)

# Input and output videos
input_video = r'D:\mecze MJO dron\cutVersion.mp4'
output_video = 'results_detected_and_tracked.mp4'

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
CONFIDENCE_LEVEL = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    tracked_boxes = []
    confidences = []
    classes_list = []  # We'll store class IDs for each detection

    # Filter and prepare detections for Deep SORT
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if cls_id in [PLAYER_CLASS_ID, BALL_CLASS_ID]:
            tracked_boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            classes_list.append(cls_id)

    # After processing ALL detections for the frame:
    if len(tracked_boxes) > 0:
        tracked_boxes_new = np.array(tracked_boxes)
        confidences_new = np.array(confidences)
        classes_new = np.array(classes_list)
        outputs, mask_outputs = deepsort.update(tracked_boxes_new, confidences_new, classes_new, frame)
    else:
        deepsort.increment_ages()
        outputs = []
        mask_outputs = []

    print("Outputs:", outputs)
    # outputs should be in format: [[x1, y1, x2, y2, track_id, class_id], ...] if your fork returns class_id

    if len(outputs) > 0:
        print('Outputs more than 0')
        for out_det in outputs:
            # out_det is something like [x1, y1, x2, y2, track_id, class_id]
            x1, y1, x2, y2, class_id, track_id = out_det
            class_id = int(class_id)
            if class_id == PLAYER_CLASS_ID:
                color = (0, 255, 0)  # Green for player
                label = f"Player {track_id}"
            elif class_id == BALL_CLASS_ID:
                color = (0, 0, 255)  # Red for ball
                label = f"Ball {track_id}"
            else:
                color = (255, 255, 255)
                label = f"ID {track_id}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out.write(frame)
    cv2.imshow("Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
