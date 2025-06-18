import string
import easyocr
from paddleocr import PaddleOCR
import numpy as np
import cv2
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHAR_TO_INT = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
INT_TO_CHAR = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
THRESHOLD_VALUE = 64
OCR_SCORE_THRESHOLD = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]

# Initialize OCR engines
reader = easyocr.Reader(['en'], gpu=True)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'
)

def load_models(coco_model_path: str, license_plate_model_path: str):
    """Load YOLO models for vehicle and license plate detection."""
    coco_model = YOLO(coco_model_path)
    license_plate_detector = YOLO(license_plate_model_path)
    return coco_model, license_plate_detector

def license_complies_format(text: str) -> bool:
    """Check if the license plate text matches the expected format."""
    if len(text) != 6:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in INT_TO_CHAR) and \
       (text[1] in string.ascii_uppercase or text[1] in INT_TO_CHAR) and \
       (text[2] in string.ascii_uppercase or text[2] in INT_TO_CHAR) and \
       (text[3] in string.digits or text[3] in CHAR_TO_INT) and \
       (text[4] in string.digits or text[4] in CHAR_TO_INT) and \
       (text[5] in string.digits or text[5] in CHAR_TO_INT):
        return True
    return False

def format_license(text: str) -> str:
    """Convert license plate text using character mappings."""
    mapping = [
        INT_TO_CHAR, INT_TO_CHAR, INT_TO_CHAR,
        CHAR_TO_INT, CHAR_TO_INT, CHAR_TO_INT
    ]
    return ''.join(mapping[i].get(text[i], text[i]) for i in range(6))

def ocr_easyocr(license_plate_crop: np.ndarray):
    """Run EasyOCR on the cropped license plate image."""
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def ocr_paddle(license_plate_crop: np.ndarray):
    """Run PaddleOCR on the cropped license plate image."""
    results = []
    try:
        ocr_result = ocr.predict(license_plate_crop)
        for res in ocr_result:
            data = getattr(res, 'res', None) or res.get('res', None) if isinstance(res, dict) else res
            if isinstance(data, dict) and 'rec_texts' in data and 'rec_scores' in data:
                for text, score in zip(data['rec_texts'], data['rec_scores']):
                    cleaned_text = text.strip()
                    if cleaned_text:
                        results.append((cleaned_text, score))
        for text, score in results:
            cleaned_text = text.upper().replace(' ', '')
            if license_complies_format(cleaned_text):
                return format_license(cleaned_text), score
    except Exception as e:
        logger.warning(f"PaddleOCR error: {e}")
    return None, None

def process_frame(
    frame: np.ndarray,
    coco_model,
    license_plate_detector,
    vehicles,
    mot_tracker,
    top_line_pos: int,
    bottom_line_pos: int,
    frame_nmr: int = 0,
    ocr_engine: str = 'easyocr'
):
    """Process a single frame for detection and OCR."""
    results = {frame_nmr: {}}
    detections = coco_model(frame)[0]
    detections_ = [detection[:5] for detection in detections.boxes.data.tolist() if int(detection[5]) in vehicles]
    try:
        track_ids = mot_tracker.update(np.asarray(detections_))
    except Exception as e:
        logger.warning(f"Tracking error: {e}")
        return results, [(frame_nmr, np.array(detections_).shape, str(e))]
    for track in track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track
        # Ensure bounding box is within frame
        xcar1, ycar1 = max(0, int(xcar1)), max(0, int(ycar1))
        xcar2, ycar2 = min(frame.shape[1], int(xcar2)), min(frame.shape[0], int(ycar2))
        vehicle_crop = frame[ycar1:ycar2, xcar1:xcar2]
        license_plates = license_plate_detector(vehicle_crop)[0]
        for license_plate in license_plates.boxes.data.tolist():
            lx1, ly1, lx2, ly2, score, _ = license_plate
            x1, y1 = int(xcar1 + lx1), int(ycar1 + ly1)
            x2, y2 = int(xcar1 + lx2), int(ycar1 + ly2)
            # Ensure license plate crop is within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            preprocessed_plate_img = None
            license_plate_text = None
            license_plate_text_score = None
            if top_line_pos < y1 < bottom_line_pos and top_line_pos < y2 < bottom_line_pos:
                license_plate_crop = frame[y1:y2, x1:x2, :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
                preprocessed_plate_img = license_plate_crop_thresh
                if ocr_engine.lower() == 'paddle':
                    license_plate_text, license_plate_text_score = ocr_paddle(license_plate_crop)
                else:
                    license_plate_text, license_plate_text_score = ocr_easyocr(license_plate_crop_thresh)
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score,
                    'preprocessed_plate_img': preprocessed_plate_img
                }
            }
    return results, []

def draw_border(
    img: np.ndarray,
    top_left: tuple,
    bottom_right: tuple,
    color=(0, 255, 0),
    thickness=10,
    line_length_x=200,
    line_length_y=200
):
    """Draw a border around the detected object."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def is_valid_ocr_result(
    ocr_text: str,
    ocr_score: float,
    last_valid_ocr: dict = None,
    car_id: int = None
) -> bool:
    """Check if the OCR result is valid and should be updated."""
    if not (ocr_text and isinstance(ocr_text, str) and ocr_score and ocr_score > OCR_SCORE_THRESHOLD):
        return False
    if last_valid_ocr and car_id is not None and car_id in last_valid_ocr:
        _, prev_score = last_valid_ocr[car_id]
        if prev_score > ocr_score:
            return False
    return True

def draw_ocr_text(
    frame: np.ndarray,
    text: str,
    score: float,
    position: tuple,
    font_scale: float = 1.2,
    font_thickness: int = 3
):
    """Draw OCR text with background at the given position."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    full_text = f"{text}"
    if score is not None:
        full_text += f" ({score:.2f})"
    (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, font_thickness)
    text_bg_height = text_height + 10
    text_bg_width = text_width + 20
    x, y = position
    text_bg_top = max(y - text_bg_height, 0)
    text_bg_left = max(x - text_bg_width // 2, 0)
    text_bg_bottom = text_bg_top + text_bg_height
    text_bg_right = text_bg_left + text_bg_width
    cv2.rectangle(frame, (text_bg_left, text_bg_top), (text_bg_right, text_bg_bottom), (255,255,255), -1)
    text_x = text_bg_left + 10
    text_y = text_bg_top + text_height + 5
    cv2.putText(frame, full_text, (text_x, text_y), font, font_scale, (0,0,255), font_thickness)
    return text_bg_height

def draw_preprocessed_plate(
    frame: np.ndarray,
    pre_img: np.ndarray,
    position: tuple,
    target_height: int = 60
):
    """Draw the preprocessed plate image at the given position."""
    if pre_img is None:
        return 0
    h, w = pre_img.shape
    scale = target_height / h
    target_width = int(w * scale)
    pre_img_resized = cv2.resize(pre_img, (target_width, target_height))
    pre_img_bgr = cv2.cvtColor(pre_img_resized, cv2.COLOR_GRAY2BGR)
    x, y = position
    pre_img_top = max(y - target_height, 0)
    pre_img_left = max(x - target_width // 2, 0)
    pre_img_bottom = pre_img_top + target_height
    pre_img_right = pre_img_left + target_width
    cv2.rectangle(frame, (pre_img_left, pre_img_top), (pre_img_right, pre_img_bottom), (255,255,255), -1)
    frame[pre_img_top:pre_img_bottom, pre_img_left:pre_img_right] = pre_img_bgr
    return target_height

def draw_detection_boxes(
    frame: np.ndarray,
    car_bbox: list,
    lp_bbox: list
):
    """Draw bounding boxes for car and license plate."""
    xcar1, ycar1, xcar2, ycar2 = map(int, car_bbox)
    x1, y1, x2, y2 = map(int, lp_bbox)
    draw_border(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 6, line_length_x=40, line_length_y=40)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return (x1, y1, x2, y2), (xcar1, ycar1, xcar2, ycar2)

def draw_detection_lines(
    frame: np.ndarray,
    top_line_pos: int,
    bottom_line_pos: int
):
    """Draw horizontal lines marking the OCR region."""
    cv2.line(frame, (0, top_line_pos), (frame.shape[1], top_line_pos), (255, 0, 0), 2)
    cv2.line(frame, (0, bottom_line_pos), (frame.shape[1], bottom_line_pos), (255, 0, 0), 2)

def visualize_detection(
    frame: np.ndarray,
    car_info: dict,
    lp_info: dict,
    last_valid_ocr: dict,
    margin: int = 10
):
    """Draw detection results, preprocessed plate, and OCR text."""
    car_bbox = car_info['bbox']
    lp_bbox = lp_info['bbox']
    pre_img = lp_info.get('preprocessed_plate_img')
    ocr_text = lp_info.get('text')
    ocr_score = lp_info.get('text_score')
    (x1, y1, x2, y2), _ = draw_detection_boxes(frame, car_bbox, lp_bbox)
    car_id = car_info.get('id')
    if car_id is not None:
        if is_valid_ocr_result(ocr_text, ocr_score, last_valid_ocr, car_id):
            last_valid_ocr[car_id] = (ocr_text, ocr_score)
        elif car_id in last_valid_ocr:
            ocr_text, ocr_score = last_valid_ocr[car_id]
        else:
            ocr_text, ocr_score = None, None
    center_x = (x1 + x2) // 2
    if pre_img is not None:
        plate_height = draw_preprocessed_plate(frame, pre_img, (center_x, y1 - margin))
        draw_ocr_text(frame, ocr_text, ocr_score, (center_x, y1 - plate_height - margin))
    else:
        draw_ocr_text(frame, ocr_text, ocr_score, (center_x, y1 - margin))