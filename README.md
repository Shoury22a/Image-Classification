Plant Leaf Image Classification using Traditional Machine Learning
Objective
Classify plant leaf images into one of the following four categories:

Healthy

Multiple Diseases

Rust

Scab

This project uses handcrafted features and traditional machine learning algorithms (SVM, Random Forest, Gradient Boosting) instead of deep learning.

Dataset
Files Used: train.csv, test.csv, sample_submission.csv, and a folder named images/

Input: JPG images of plant leaves

Labels: Multi-class classification labels (healthy, rust, scab, multiple_diseases)

Features Extracted
Raw pixel values

Color histograms

Image dimensions (height, width)

Note: Texture and shape-based handcrafted features are not explicitly used.

Models Considered
1. Support Vector Machine (SVM)
Linear Kernel: Accuracy = 0.51

RBF Kernel: Accuracy = 0.55

Polynomial Kernel: Accuracy = 0.56 (Best among SVM kernels)

2. Gradient Boosting Classifier
Accuracy = 0.79

3. Random Forest Classifier
Accuracy = 0.80

Weighted F1 Score = 0.77

Final Model: Random Forest
Selected based on its balanced performance and highest overall accuracy.

Accuracy: 0.80

F1 Score (Weighted): 0.77

Saved as .pkl file for future inference and web integration.

Libraries Used
OpenCV – Image loading and processing

NumPy, Pandas – Data handling

Matplotlib, Seaborn, Plotly – Visualization

scikit-learn – ML algorithms and evaluation

tqdm – Progress bar tracking

Pillow – Image metadata and size analysis

Output
Model .pkl files generated for SVM, Random Forest, and Gradient Boosting

Designed for use with a Streamlit web app to predict the condition of a plant leaf from an uploaded image

How to Run the Web Application
To run the Streamlit app locally, follow these steps:

bash
Copy
Edit
# 1. Clone the repository
git clone https://github.com/Shoury22a/Image-Classifiication-

# 2. Navigate into the project directory
cd Image-Classification

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the web app
streamlit run app1.py
