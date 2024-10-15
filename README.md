# Customized-Face-Recognising-Deep-Model-Project

## Introduction:
The field of computer vision has seen significant advancements, particularly in object detection and facial recognition tasks. Among these, face detection is a crucial application with broad use cases in security systems, identity verification, and human-computer interaction. This project focuses on developing a tailored face detection system that leverages machine learning algorithms, deep learning architectures, and computer vision libraries like OpenCV. The model is trained from scratch, offering flexibility for adaptation to specific datasets and use cases, including detecting objects beyond faces if required.

## Scope of Project:

### Data Preprocessing:
Acquisition of a specialized dataset, including images under diverse lighting conditions.
Application of data augmentation techniques like rotation, zoom, and cropping to enrich the dataset.

### Model Architecture:
VGG16: A pre-trained convolutional neural network (CNN) architecture is employed as the base for the face detection model.

### Model Training:
Training is conducted from scratch on a specialized dataset.
Loss Function: Binary cross-entropy for classification tasks, while mean squared error (MSE) is used for bounding box regression.

### Optimizer:
Adaptive optimizers such as Adam or RMSProp are employed.

### Real-Time Detection:
The model is capable of real-time detection via a video feed, predicting face locations and highlighting them with bounding boxes.

### Performance Metrics:
Model performance is evaluated using Precision, Recall, F1 Score, and Intersection over Union (IoU) to measure bounding box accuracy.

### Challenges Addressed:
Variability in lighting, pose, occlusion, and expression.
Custom object detection beyond faces with retraining on relevant datasets.

## Features : 

### Real-time Face Detection:
Detect faces via webcam in real-time.

### Custom Object Detection: 
Capable of detecting other objects (e.g., helmets, masks) with retraining.

### Augmentation: 
Enriched training data through data augmentation techniques.

### High Accuracy: 
Uses pre-trained VGG16 architecture, customized to improve accuracy.

### Flexible Deployment: 
Can be deployed in various environments, including desktop or cloud-based systems.

### Optimized for Low Latency:
Efficient real-time performance in both detection and tracking.

## Installation

Clone the Repository:
```
git clone https://github.com/yourusername/FaceRecognitionModel.git
cd FaceRecognitionModel
```

Create Virtual Environment:
```
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

Install Dependencies:
```
pip install -r requirements.txt
```

## Usage :

### Image Collection:
Capture images using a webcam via OpenCV.
```
import os
import cv2
import uuid
import time
cap = cv2.VideoCapture(0)
for imgnum in range(30):
    ret, frame = cap.read()
    imgname = os.path.join('data', 'images', f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()
```

### Data Augmentation:
Apply augmentations using Albumentations.
```
import albumentations as alb
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2)
])
```

### Model Building:
Use VGG16 for face detection and bounding box regression.
```
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense

def build_model():
    input_layer = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Head
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding Box Regression Head
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    model = Model(inputs=input_layer, outputs=[class2, regress2])
    return model
```

### Model Training and Compilation:
Compile the model using custom loss functions for both classification and bounding box regression.
```
facetracker = FaceTracker(build_model())
facetracker.compile(
    opt=tf.keras.optimizers.Adam(), 
    classloss=tf.keras.losses.BinaryCrossentropy(),
    localizationloss=localization_loss
)
```
## System Architecture:
![{0E17491B-3CE6-4C3B-9DC2-CB5B166B5436}](https://github.com/user-attachments/assets/4aa53433-5e31-4b2a-bab8-849ec3b61907)

## Output:
The project demonstrated real-world object detection capabilities by leveraging a webcam to capture live video feeds. Utilizing a deep learning model trained on a diverse dataset, the system identified and classified objects in real time.
![{70F4961A-500D-43E7-84B0-267AFDAF6584}](https://github.com/user-attachments/assets/20479665-2c00-44d9-ae89-05a589e4b618)

## Result:
