# 🏢 Employee Retention Predictor  

This project predicts whether an employee is likely to **stay or leave a company** based on their profile information.  

It combines a **machine learning model** (trained in `main.ipynb`) with an interactive **Streamlit web application (`app.py`)**.  

The web app allows HR managers and recruiters to input employee data and get an instant prediction on whether the employee is likely to stay or seek a job change.  

---

## 🚀 Features  

✅ **ML Model for Employee Retention Prediction**  
✅ **Interactive Streamlit Web App**  
✅ **Manual Encoding for Categorical Variables**  
✅ **Pre-trained Model & Scaler included** (`grid_model.pkl`, `scaler.pkl`)  
✅ **Clear Feature Encoding Reference inside the app**  
✅ **Lightweight & Easy to Deploy**  

---

## 📸 App Preview  

Here’s how the Streamlit app looks:  

### 🖼 Employee Information Form  
![Employee Info](Streamlit%20view.png)  

### 🖼 Prediction: Employee Likely to Leave  
![Prediction Leave](Streamlit%20leave.png)  

### 🖼 Prediction: Employee Likely to Stay  
![Prediction Stay](Streamlit%20stay.png)  

---

## 🔍 How It Works  

1️⃣ **User enters employee information**  
- City Development Index  
- Relevant Experience  
- Education Level  
- Enrolled University  
- Major Discipline  
- Years of Experience  
- Company Size & Type  
- Years since last job change  

2️⃣ **Input is encoded & scaled**  
- The app applies **manual encoding** for categorical fields  
- Features are **scaled using StandardScaler** (same as model training)  

3️⃣ **Model makes a prediction**  
- `0` → **Employee likely to stay**  
- `1` → **Employee likely to leave**  

4️⃣ **Result is displayed in the UI**  
- Green message for **likely to stay**  
- Red warning for **likely to leave**  

---

## 🧠 Model Used  

- **Algorithm** → Random Forest Classifier  
- **Hyperparameter Tuning** → GridSearchCV for best parameters  
- **Comparison** → Tested with XGBoost but Random Forest gave better accuracy  
- **Evaluation** → Average AUC ~ **0.80**  

The training process is in `main.ipynb`, which includes:  
✅ Data Cleaning & Preprocessing  
✅ Encoding categorical features  
✅ Handling imbalance (without SMOTE since it reduced performance)  
✅ Model Training with GridSearch RandomForest  
✅ Model Saving (`grid_model.pkl`) & Scaler Saving (`scaler.pkl`)  

---
