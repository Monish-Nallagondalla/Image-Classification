# Sports Celebrity Image Classification: Data Cleaning

This project focuses on classifying images of sports celebrities (and other notable figures) using a machine learning pipeline. The initial step involves data cleaning and preprocessing, where raw images are processed to detect and crop faces with at least two visible eyes. Features are then extracted using wavelet transforms to prepare the data for training a classifier.

Special thanks to **Debjyoti Paul**, a data scientist friend at Amazon, for his guidance on this project.

## Project Overview

- **Objective**: Classify images of celebrities by preprocessing them to detect and crop facial regions, then extracting features for machine learning.
- **Dataset**: Custom dataset of celebrity images stored in the `dataset/` directory.
- **Preprocessing**: Face detection using Haar cascades, eye detection, cropping, and wavelet transform for feature extraction.
- **Model**: SVM with RBF kernel (tuned via GridSearchCV) as the primary classifier.
- **Evaluation**: Confusion matrix, precision, recall, and F1-score.

## Dataset

The dataset consists of images organized in subdirectories under `dataset/`, each named after a celebrity (e.g., `lionel_messi`, `maria_sharapova`, etc.). The preprocessing pipeline crops facial regions and saves them in `dataset/cropped/` for model training. Example celebrities include:
- Lionel Messi
- Maria Sharapova
- Roger Federer
- Serena Williams
- Virat Kohli

Sample image preprocessing:
- **Input**: Raw image (e.g., `sharapova1.jpg` - 555x700x3).
- **Output**: Cropped facial region (if two eyes are detected).

## Prerequisites

To run this project, you need the following Python libraries:
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `pywavelets` (PyWavelets)
- `scikit-learn`
- `seaborn` (for visualization)

Install the dependencies using pip:
```bash
pip install numpy opencv-python matplotlib pywavelets scikit-learn seaborn
```

Additionally, download the Haar cascade files for face and eye detection from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place them in `opencv/haarcascades/`:
- `haarcascade_frontalface_default.xml`
- `haarcascade_eye.xml`

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Monish-Nallagondalla/Image-Classification.git
   cd Image-Classification
   ```

2. **Prepare the Dataset**:
   - Place raw celebrity images in subdirectories under `dataset/` (e.g., `dataset/lionel_messi/`).
   - Ensure the `opencv/haarcascades/` directory contains the Haar cascade XML files.

3. **Run the Code**:
   - Open the `.ipynb` files in Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Execute the cells in the following notebooks:
     1. Data cleaning and preprocessing (face detection, cropping, wavelet transform).
     2. Model training and evaluation (SVM, GridSearchCV).

4. **Output**:
   - Cropped images saved in `dataset/cropped/`.
   - Visualizations of raw and cropped images.
   - Model performance metrics (e.g., accuracy, confusion matrix).

## Code Breakdown

### 1. Data Cleaning and Preprocessing
- **Face Detection**: Uses `haarcascade_frontalface_default.xml` to detect faces in grayscale images.
- **Eye Detection**: Uses `haarcascade_eye.xml` to ensure at least two eyes are visible.
- **Cropping**: Crops the facial region if two eyes are detected, discarding obstructed or unclear images.
- **Wavelet Transform**: Applies a 5-level Daubechies wavelet transform (`db1`) to extract edge features.

Key function:
```python
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
```

- **Output**: Cropped images saved in `dataset/cropped/<celebrity_name>/` (e.g., `messi1.png`).

### 2. Feature Extraction
- **Raw Images**: Resized to 32x32 pixels (3072 features: 32*32*3).
- **Wavelet Transform**: Resized to 32x32 pixels (1024 features: 32*32).
- **Combined Features**: Stacked vertically to form a 4096-dimensional vector per image.

### 3. Model Training
- **Pipeline**: StandardScaler + SVM (RBF kernel, C=10).
- **GridSearchCV**: Evaluates SVM, Random Forest, and Logistic Regression with various parameters.
- **Best Model**: SVM with RBF kernel (accuracy: ~75%).

### 4. Evaluation
- **Confusion Matrix**:
```
[[21  1  1  0  3  0]
 [ 0 12  2  0  2  2]
 [ 1  2 30  0  1  1]
 [ 1  0  0 24  2  1]
 [ 2  0  2  1 19  2]
 [ 2  3  4  0  0  4]]
```
- **Classification Report**:
```
              precision    recall  f1-score   support
     0 (Elon Musk)     0.78      0.81      0.79        26
     1 (Kanye West)    0.67      0.67      0.67        18
     2 (Megan Fox)     0.77      0.86      0.81        35
     3 (Modi)          0.96      0.86      0.91        28
     4 (Putin)         0.70      0.73      0.72        26
     5 (Travis Scott)  0.40      0.31      0.35        13
    accuracy                           0.75       146
```

## Results

- **Preprocessing**: Successfully cropped images with two visible eyes, discarding obstructed ones.
- **Model Performance**: Achieved ~75% accuracy with SVM (RBF kernel). Modi (class 3) has the highest precision (0.96), while Travis Scott (class 5) has the lowest recall (0.31).
- **Visualizations**: Heatmap of confusion matrix highlights classification performance.

## Future Improvements

- Expand the dataset with more sports celebrities and images.
- Implement data augmentation (e.g., rotation, flipping) to increase robustness.
- Experiment with deep learning models (e.g., CNNs) for better feature extraction.
- Fine-tune wavelet transform parameters or explore other feature extraction methods.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Debjyoti Paul** for his invaluable insights and support.
- OpenCV documentation: [Face Detection Tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html).
- Built with Python, OpenCV, PyWavelets, scikit-learn, and matplotlib.

## Repository

[GitHub Repository](https://github.com/Monish-Nallagondalla/Image-Classification)
```

### Instructions
1. Copy the entire code block above.
2. Open or create a `README.md` file in your GitHub repository (`https://github.com/Monish-Nallagondalla/Image-Classification`).
3. Paste the code into the file.
4. Ensure the dataset structure (`dataset/` and `opencv/haarcascades/`) matches the paths in the code.
5. Save and commit the file to your repository.
