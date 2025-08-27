# 🫀 Heart Disease Predictor

A machine learning project that predicts the likelihood of **heart disease** using patient medical attributes.
This project uses **Logistic Regression** and **Random Forest Classifier**, along with feature engineering and scaling, to build an accurate classifier.

---

## 📂 Project Structure

```
disease Predictor.ipynb   # Main Jupyter Notebook with all steps
heart-disease/            # Dataset folder (from Kaggle)
rf_model.pkl              # Saved Random Forest model
heart_scaler.pkl          # Saved StandardScaler for input features
Heart_user_temp.csv       # Template for user input
```

---

## 🚀 Steps Covered

### **Day 1 – Dataset Setup**

* Downloaded dataset from Kaggle using `kaggle.json`.
* Loaded dataset into pandas DataFrame.
* Checked dataset attributes and missing values.
* Filled numeric missing values with **mean**.
* Filled categorical missing values with **mode**.

---

### **Day 2 – Data Preprocessing & Visualization**

* Converted categorical columns into numerical (One-Hot Encoding).
* Prepared feature set (`X`) and target variable (`y`).
* Visualized:

  * Histograms of features (`matplotlib`).
  * Heatmap of correlations (`seaborn`).

---

### **Day 3 – Model Training**

* Split data into training (80%) and testing (20%).
* Standardized features using **StandardScaler**.
* Trained **Logistic Regression** model:

  * Achieved accuracy \~84%.
  * Evaluated with **precision, recall, f1-score**.

---

### **Day 4 – Advanced Models**

* Trained a **Random Forest Classifier**:

  * Achieved accuracy \~87.5%.
* Plotted **feature importance** to see which attributes influence predictions most.
* Saved the trained model and scaler with **joblib**.

---

### **Day 5 – User Input & Prediction**

* Created `Heart_user_temp.csv` as a **user input template**.
* Allowed users to upload their own dataset (`heart_dataset.csv`).
* Applied the same preprocessing (encoding, scaling).
* Loaded trained Random Forest model (`rf_model.pkl`).
* Predicted whether each user has **Heart Disease (1) or Not (0)**.

---

## 📊 Example Output

```
Logistic Regression Accuracy: 0.842
Random Forest Accuracy: 0.875
```

Sample prediction output:

| age | trestbps | chol | thalch | oldpeak | slope | sex\_Male | Heart Disease |
| --- | -------- | ---- | ------ | ------- | ----- | --------- | ------------- |
| 55  | 120      | 200  | 140    | 1.2     | 0     | 0         | 0             |
| 62  | 130      | 220  | 150    | 2.0     | 1     | 1         | 1             |

---

## ⚙️ Installation & Usage

1. Clone the repository / open in Google Colab.
2. Authenticate Kaggle and download the dataset:

   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d redwankarimsony/heart-disease-data -p /content/heart-disease --unzip
   ```
3. Install dependencies:

   ```bash
   pip install kaggle scikit-learn pandas matplotlib seaborn joblib
   ```
4. Run all cells in **disease Predictor.ipynb**.
5. Upload `heart_dataset.csv` (your custom input).
6. Get predictions in the notebook.

---

## 📌 Key Learnings

* Data preprocessing (handling NaNs, encoding categorical features).
* Train-test split and feature scaling.
* Logistic Regression vs Random Forest performance.
* Model evaluation with accuracy, precision, recall, f1-score.
* Saving & loading models with joblib.
* Allowing real user data input for prediction.

---

## 📈 Future Improvements

* Deploy as a **Flask/Django web app**.
* Add **Streamlit UI** for user-friendly predictions.
* Hyperparameter tuning for better accuracy.
* Cross-validation for model robustness.


