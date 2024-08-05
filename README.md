# YOLOv8 Object Detection and Sheet Counting App

This Streamlit app leverages the YOLOv8 model for object detection and calculates the number of sheets in a stack based on detected bounding box dimensions. The app allows users to upload an image, detects objects, and provides an estimate of the number of sheets if applicable.

## Features

- **Image Upload**: Users can upload images in `jpg`, `jpeg`, or `png` formats.
- **Object Detection**: Utilizes the YOLOv8 model to detect objects in the uploaded image and draw bounding boxes around them.
- **Sheet Counting**: Calculates the number of sheets in a stack using the height of the detected stack and the average thickness of a sheet.

## Requirements

- **Python 3.x**
- **Streamlit**: Web app framework
- **Ultralytics**: For YOLOv8 model
- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **Pillow**: For image manipulation

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/yolov8-sheet-counting-app.git
   cd yolov8-sheet-counting-app
   ```
2. ***Install the Required Libraries***:
Run the following command to install the necessary packages:

```bash
pip install streamlit ultralytics opencv-python-headless numpy pillow
```
3. ***Run the Application***:
Start the Streamlit app by running:

```bash
streamlit run app.py
```
The app will be available in your web browser at http://localhost:8501.

## Usage
Upload an Image: Click on the "Choose an image..." button to upload your image.
***View Detection Results***: The app will display the uploaded image and an annotated version with detected objects.
***Sheet Counting***: If a stack of sheets is detected, the app will calculate and display the estimated number of sheets.
## Example
Upload an Image: For example, upload an image of a stack of sheets.
Detection and Counting: The app will process the image, display the detected objects, and output the estimated number of sheets based on the detected bounding box height.
## Code Overview
Loading the Model: The YOLOv8 model is loaded from the specified path (best.pt).
Image Upload and Processing: Users upload an image which is then processed for object detection.
Object Detection and Annotation: Detected objects are annotated with bounding boxes and class labels.
Sheet Counting: The app calculates the number of sheets in a stack using the detected height and a known average sheet thickness of 12.5 units.


### Key Sections

- **Features**: Describes what the app does, including image upload, object detection, and sheet counting.
- **Requirements**: Lists the necessary Python libraries and packages.
- **Installation**: Provides steps to clone the repository, install dependencies, and run the app.
- **Usage**: Explains how to use the app once it’s running.
- **Example**: Demonstrates typical use case scenarios.
- **Code Overview**: Briefly explains the core functionalities implemented in the code.

