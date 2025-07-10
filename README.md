# Plant Leaf Image Classification using Traditional Machine Learning

## Objective  
Classify plant leaf images into one of the following four categories:  
- Healthy  
- Multiple Diseases  
- Rust  
- Scab  

This project uses handcrafted features and traditional machine learning algorithms (SVM, Random Forest, Gradient Boosting) instead of deep learning approaches.

---

## Dataset  
- **Files Used**: `train.csv`, `test.csv`, `sample_submission.csv`, and a folder named `images/`
- **Input**: JPG images of plant leaves  
- **Labels**: Multi-class classification — `healthy`, `rust`, `scab`, and `multiple_diseases`

---

## Features Extracted  
- Raw pixel values  
- Color histograms  
- Image dimensions (height and width)  

> Note: Texture and shape-based handcrafted features are **not** explicitly used.

---

## Models Considered

### 1. Support Vector Machine (SVM)
| Kernel      | Accuracy |
|-------------|----------|
| Linear      | 0.51     |
| RBF         | 0.55     |
| Polynomial  | **0.56** (Best among SVM kernels)

### 2. Gradient Boosting Classifier
- Accuracy: 0.79

### 3. Random Forest Classifier
- Accuracy: **0.80**
- Weighted F1 Score: **0.77**

---

## Final Model: Random Forest
Chosen due to its balanced performance and highest accuracy.  
- **Accuracy**: 0.80  
- **Weighted F1 Score**: 0.77  
- Model saved as `.pkl` for deployment and inference.

---

## Libraries Used  
- `OpenCV` – for image loading  
- `NumPy`, `Pandas` – for data handling  
- `Matplotlib`, `Seaborn`, `Plotly` – for visualizations  
- `scikit-learn` – for ML models and evaluation  
- `tqdm` – for progress tracking  
- `Pillow` – for analyzing image dimensions

---

## Output  
- Trained model `.pkl` files generated for each algorithm  
- Ready for integration with a Streamlit web application to classify plant leaf conditions from uploaded images

---

## How to Run the Web Application  

To run the Streamlit app locally, follow these steps:

```bash
# Step 1: Clone the repository
git clone https://github.com/Shoury22a/Image-Classifiication-

# Step 2: Navigate to the project directory
cd Image-Classification

# Step 3: Install required dependencies
pip install -r requirements.txt

# Step 4: Launch the Streamlit app
streamlit run app1.py
