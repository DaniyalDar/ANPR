import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models')))

from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from utils import (
    process_frame, load_models, draw_detection_lines, visualize_detection
)

# Select OCR engine: 'easyocr' or 'paddle'
OCR_ENGINE = 'easyocr'

# Load YOLO models for vehicle and license plate detection
coco_model, license_plate_detector = load_models(
    '/Models/yolov12m.pt',
    '/Models/Lisence_Plate.pt'
)

# Open input video
input_video_path = 'b.mp4'
cap = cv2.VideoCapture(input_video_path)

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_annotated.mp4', fourcc, fps, (width, height))

# Vehicle class IDs for detection
vehicles = [2, 3, 5, 7]

# Define OCR region lines
top_line_pos = 990
bottom_line_pos = 1800

mot_tracker = Sort()
frame_nmr = 0
last_valid_ocr = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results, _ = process_frame(
        frame, coco_model, license_plate_detector, vehicles, mot_tracker,
        top_line_pos, bottom_line_pos, frame_nmr, ocr_engine=OCR_ENGINE
    )

    draw_detection_lines(frame, top_line_pos, bottom_line_pos)

    for car_id in results.get(frame_nmr, {}):
        car_info = results[frame_nmr][car_id]['car']
        lp_info = results[frame_nmr][car_id]['license_plate']
        car_info['id'] = car_id
        visualize_detection(frame, car_info, lp_info, last_valid_ocr)

    out.write(frame)
    frame_nmr += 1

cap.release()
out.release()