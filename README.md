# Face-Recognition
A face recognition model that detects, analyzes, and matches human faces using the K-Nearest Neighbors (KNN) algorithm, ensuring reliable identification for authentication and personalization.
Absolutely! Here's an explanation based on the provided data:

### Project Workflow:
1. **Data Collection**:
    - **File**: `project.py`
    - **Purpose**: Capture 30-40 images of different people.
    - **Process**: This script captures images via a webcam and saves them as data files for each person. These images are used for training a face recognition model.

2. **Prediction**:
    - **File**: `test.py`
    - **Purpose**: Make predictions using the captured data.
    - **Process**: This script uses the stored images to predict the identity of people in new images or video frames.

### Explanation of the Provided Data:
- **Images**: The dataset consists of images captured from different people. These images are stored in the dataset directory.
- **Labels**: Each image is associated with a label that identifies the person. Labels are stored in `.npy` files.

### How It Works:
1. **Running `project.py`**:
    - Initialize the webcam and capture multiple images of each person.
    - Store these images in the dataset directory for future use.

2. **Running `test.py`**:
    - Load the saved images and their labels.
    - Use these images to recognize faces in new images or video streams.

The two scripts work together to first collect and then use facial data for recognition purposes.
### Folder Structure:
face-recognition/
    -data/
         - //all the .npy files will store here//
    -project.py
    -test.py


