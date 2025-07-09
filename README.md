Plant Leaf Image Classification using Traditional Machine Learning
Objective
Classify plant leaf images into one of four categories:

Healthy

Multiple Diseases

Rust

Scab

This project uses handcrafted features and traditional ML algorithms (SVM, Random Forest, Gradient Boosting) instead of deep learning.

Dataset
Files Used: train.csv, test.csv, sample_submission.csv, and a folder named images/

Input: JPG images of plant leaves

Labels: Multi-class labels from the dataset (healthy, rust, etc.)

Features Extracted
Raw pixel values

Color histograms

Image size and dimensions (height, width)

Texture and shape-based handcrafted features are not used explicitly in this notebook.

Models Used
Support Vector Machine (SVM) with polynomial kernel

Random Forest Classifier

Gradient Boosting Classifier

SVM was noted to perform best among the models during experimentation.

Performance
Approximate accuracy for each model was observed around 32%.

No formal train/test split or cross-validation analysis was visible — the notebook uses the train.csv entries for modeling and test.csv for unseen predictions.

Tech Stack
Python

Libraries:

OpenCV for image loading

NumPy, Pandas for data handling

Matplotlib, Seaborn, Plotly for visualization

scikit-learn for ML models

tqdm for progress tracking

Pillow for image dimension analysis

Output
.pkl files generated for each model for future inference.

Intended for integration with a Streamlit web app that takes an image and shows the predicted leaf condition.
