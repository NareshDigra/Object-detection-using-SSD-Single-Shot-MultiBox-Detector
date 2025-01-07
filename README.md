# Object detection using SSD (Single Shot MultiBox Detector)
This repository demonstrates object detection using MobileNet SSD (Single Shot MultiBox Detector). It is capable of detecting and labeling objects in real-time from a webcam or in pre-recorded videos.

# Features

```Real-time Object Detection: Detect objects from live video feed or pre-recorded video.```

```MobileNet SSD: Utilizes a lightweight yet powerful model for efficient object detection.```

```Customizable Confidence Threshold: Adjustable confidence level to filter predictions.```

```GPU Acceleration: Option to leverage CUDA for faster detection (if a compatible GPU is available).```

# Requirements

## Dependencies

### Make sure you have the following installed:

Python 3.7+

OpenCV

NumPy

imutils

# Install the required Python packages using pip:

``` pip install numpy imutils opencv-python ```

# Model Files

## You need the following files for MobileNet SSD:

### Prototxt File: Defines the architecture of the network.

``` Path: ssd_files/MobileNetSSD_deploy.prototxt ```

### Caffe Model File: Pre-trained weights for the network.

``` Path: ssd_files/MobileNetSSD_deploy.caffemodel ```

# You can download these files from the official MobileNet SSD repository.

# Usage

1. Clone the Repository

``` git clone https://github.com/your-username/object-detection-ssd.git ```
``` cd object-detection-ssd ```

2. Run the Script

Run the detection script:

``` python object_detection.py ```

3. Key Variables in the Code

``` live_video: Set to True for live webcam feed or False to use a video file. ```

``` use_gpu: Enable CUDA acceleration by setting to True. ```

``` confidence_level: Adjust the minimum confidence for displaying detections (default is 0.5). ```

# Code Walkthrough

## Core Steps

### Load Model Files:

``` net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path) ```

### Set GPU Acceleration:

``` net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) ```
``` net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) ```

# Process Video Frames:

Resize and normalize the input frames.

Use blobFromImage to preprocess frames for the network.

# Perform Detection:

Pass preprocessed frames to the network.

Parse and display detections with bounding boxes and labels.

# Key Functions

``` cv2.dnn.blobFromImage: Prepares the image for the model by resizing, normalizing, and converting to the required format. ```

``` cv2.rectangle: Draws bounding boxes on detected objects. ```

``` cv2.putText: Displays object labels and confidence scores. ```

# Example Output:

Real-time detection with bounding boxes and labels displayed over detected objects.

# Supported Classes

## The following object categories can be detected:

```['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']```

# Notes

```Ensure your prototxt and caffemodel files are in the correct paths.```

```CUDA support requires an NVIDIA GPU and appropriate drivers installed.```

```For better performance on CPU, disable CUDA by setting use_gpu = False.```

# Author

## Naresh Kumar

