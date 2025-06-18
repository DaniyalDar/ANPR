Vehicle and License Plate Detection

This project detects vehicles and license plates in videos using YOLO models and applies Optical Character Recognition (OCR) with EasyOCR or PaddleOCR to recognize license plate text. The license plate and OCR processing are specifically designed for Pakistani number plates.

Features

- Detects vehicles using YOLO.
- Extracts and recognizes Pakistani license plates using OCR.
- Tracks vehicles across frames using the SORT algorithm.
- Annotates video frames with vehicle and license plate detections and OCR results.
- Outputs a video with visualized annotations.

Requirements

- Python 3.x
- OpenCV
- NumPy
- easyOCR
- paddleOCR
- ultralytics (for YOLO models)
- SORT (for vehicle tracking)
- torch (for Ultralytics YOLO models)

To install the required dependencies, follow these steps:

1. Set up a conda environment:

   conda create --prefix ./env python=3.10
   conda activate ./env

2. Install torch:

   pip install torch==2.2.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

3. Install additional dependencies:

   pip install opencv-python numpy easyocr paddlepaddle ultralytics sort

Project Structure

- main.py: Main script that processes the video, applies detections, and generates the output video.
- utils.py: Contains utility functions for model loading, frame processing, and OCR.

Usage

1. Clone the repository:

   git clone https://github.com/yourusername/vehicle-license-plate-detection.git
   cd vehicle-license-plate-detection

2. Place your video (e.g., b.mp4) in the project directory.

3. Ensure YOLO models are in the /Models/ directory. You can download them from the YOLOv5 releases (e.g., yolov12m.pt for vehicle detection and Lisence_Plate.pt for license plate detection).

4. Run the script:

   python main.py

5. The output video with annotations will be saved as output_annotated.mp4.

Configuration

- OCR Engine: Choose between easyocr or paddleocr by setting the OCR_ENGINE variable in main.py.
- Vehicle Classes: The current configuration detects vehicles with the following class IDs: [2, 3, 5, 7] (car, bus, truck, motorbike in COCO dataset).
- OCR Region: OCR is applied within a defined region of the video. Adjust the top_line_pos and bottom_line_pos variables in main.py to modify the region.
- Pakistani Number Plates: The license plate and OCR processing is specifically designed to work with Pakistani number plates, considering common formats and character substitutions.

