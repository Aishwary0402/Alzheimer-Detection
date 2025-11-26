# 🧠 Alzheimer's Disease Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📖 Project Overview

This project focuses on the early diagnosis and prediction of Alzheimer's disease using Machine Learning techniques. By analyzing a comprehensive set of patient health data—including demographic information, lifestyle factors, and clinical measurements—the project evaluates multiple classification models to determine the most effective approach for accurate detection.

The analysis is performed entirely within a Jupyter Notebook (`Alzheimer.ipynb`), covering the end-to-end pipeline from data cleaning to model evaluation.

## 🚀 Key Features

* **Data Preprocessing:** Robust handling of missing values, duplicate removal, and feature scaling using MinMax Scaler.
* **Exploratory Data Analysis (EDA):** Visualizations (Box plots, Histograms) to understand data distribution, outliers, and correlations.
* **Multi-Model Evaluation:** Implementation and comparison of six different machine learning algorithms:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Support Vector Machine (SVM)
    * K-Nearest Neighbors (KNN)
    * Gradient Boosting
* **Performance Metrics:** Detailed assessment using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 🛠️ Tech Stack

* **Language:** Python
* **IDE:** Jupyter Notebook
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

## 📂 Dataset Details

The analysis uses the `alzheimers_disease_data.csv` dataset, which includes the following feature categories:

1.  **Demographics:** Age, Gender, Ethnicity, Education Level.
2.  **Health Metrics:** BMI, Smoking Status, Alcohol Consumption, Physical Activity.
3.  **Clinical History:** Diabetes, Hypertension, Family History of Alzheimer's.
4.  **Clinical Measurements:** Systolic/Diastolic BP, Cholesterol Levels.
5.  **Cognitive Assessments:** MMSE (Mini-Mental State Exam), Functional Assessment scores.

**Target Variable:** `Diagnosis` (0 = No Alzheimer's, 1 = Alzheimer's Detected).

## ⚙️ Installation & Usage

To run this project locally, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Aishwary0402/Alzheimer-Detection.git](https://github.com/Aishwary0402/Alzheimer-Detection.git)
    cd Alzheimer-Detection
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. You can install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

3.  **Run the Notebook**
    Launch Jupyter Notebook to view and execute the analysis:
    ```bash
    jupyter notebook Alzheimer.ipynb
    ```

## 📊 Results & Methodology

The project follows a structured workflow:
1.  **Data Loading:** Importing the dataset and dropping irrelevant IDs (PatientID, DoctorInCharge).
2.  **Scaling:** Normalizing numerical features to ensure all models perform optimally.
3.  **Training:** Splitting data into Training and Testing sets (typically 80/20 split).
4.  **Comparison:** All six models are trained and their accuracy scores are tabulated to identify the best performer.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model accuracy or adding Deep Learning approaches (CNNs on MRI data), please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/NewModel`).
3.  Commit your changes.
4.  Open a Pull Request.

## 📜 License

This project is open-source and available for educational and research purposes.
