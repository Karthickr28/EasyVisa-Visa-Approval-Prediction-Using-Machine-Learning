# 🧾 EasyVisa — Visa Approval Prediction Using Machine Learning

## 📘 Project Overview

The **EasyVisa** project aims to automate and optimize the process of visa certification handled by the **Office of Foreign Labor Certification (OFLC)** in the United States.  
With the rapid increase in visa applications every year, manually reviewing each case has become inefficient and time-consuming.  

This project applies **machine learning classification techniques** to predict whether a visa application will be **certified** or **denied**, based on employer, employee, and job-related attributes.  
The goal is to help OFLC analysts **prioritize applications**, reduce manual workload, and improve decision-making consistency.

---

## 🎯 Business Objective

The **Office of Foreign Labor Certification (OFLC)** processes hundreds of thousands of visa applications annually.  
To address growing application volumes, **EasyVisa** was contracted to build a **data-driven system** that can:

1. Predict visa approval outcomes (Certified / Denied).  
2. Identify key factors that influence visa decisions.  
3. Provide actionable recommendations to improve approval likelihood for genuine applicants.

---

## 🧠 Problem Statement

- The OFLC processes nearly **1.7 million positions** annually for labor certification.  
- Reviewing each case manually is tedious and error-prone.  
- A **predictive ML model** can assist in shortlisting applicants who are more likely to receive visa approval.  

As a **Data Scientist** at EasyVisa, your task is to analyze the provided dataset, identify important features affecting case outcomes, and develop robust machine learning models to automate the shortlisting process.

---

## 📂 Dataset Description

The dataset includes information about **employees, employers, and job positions** submitted for visa applications.

| **Feature** | **Description** |
|--------------|----------------|
| `case_id` | Unique ID for each visa application |
| `continent` | Continent of employee origin |
| `education_of_employee` | Education qualification of the employee |
| `has_job_experience` | Whether the employee has job experience (Y/N) |
| `requires_job_training` | Whether the employee requires job training (Y/N) |
| `no_of_employees` | Number of employees in the employer's company |
| `yr_of_estab` | Year when the employer’s company was established |
| `region_of_employment` | Intended region of employment in the U.S. |
| `prevailing_wage` | Average wage for the job position |
| `unit_of_wage` | Wage unit (Hourly, Weekly, Monthly, Yearly) |
| `full_time_position` | Whether the job is full-time (Y/N) |
| `case_status` | Target variable — Visa case outcome (Certified/Denied) |

---

## 🔍 Exploratory Data Analysis (EDA)

Key EDA objectives included:
- Distribution of visa outcomes (Certified vs. Denied)  
- Education level and its impact on case status  
- Relationship between experience, training, and visa approval  
- Effect of company size (`no_of_employees`) and establishment year (`yr_of_estab`)  
- Prevailing wage and wage unit trends  
- Employment region analysis — which regions have higher approval rates  

Visualizations and summary statistics were used to uncover hidden patterns and detect potential class imbalances.

---

## ⚙️ Machine Learning Workflow

### **1️⃣ Data Preprocessing**
- Handled missing and inconsistent values  
- Encoded categorical variables (`continent`, `education_of_employee`, etc.)  
- Normalized numerical features like `prevailing_wage`  
- Split data into **train** and **test** sets  

---

### **2️⃣ Model Building (Original Data)**
Models applied:
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- AdaBoost Classifier  
- XGBoost Classifier  

Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

**Objective:** Identify baseline performance without resampling.

---

### **3️⃣ Model Building (Oversampled Data)**
- Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.  
- Re-trained all five models on the oversampled dataset.  
- Compared performance improvements vs. baseline.  

---

### **4️⃣ Model Building (Undersampled Data)**
- Applied **Random Undersampling** on the majority class.  
- Re-built and evaluated models again.  
- Compared precision-recall tradeoffs.  

---

### **5️⃣ Model Tuning and Optimization**
Selected top 3 models (based on prior results):
- Random Forest  
- XGBoost  
- Gradient Boosting  

Performed **Hyperparameter Tuning** using **RandomizedSearchCV** on key parameters:
- Number of estimators  
- Maximum depth  
- Learning rate  
- Minimum samples per leaf  

Best models selected based on **ROC-AUC** and **F1-score** performance on validation data.

---

### **6️⃣ Model Comparison and Selection**
| Model | Sampling Method | Best Metric (AUC/F1) | Remarks |
|--------|----------------|-----------------------|----------|
| Random Forest | SMOTE | 0.91 | High recall, interpretable |
| XGBoost | Original | 0.93 | Best overall performance |
| Gradient Boosting | Oversampled | 0.90 | Good precision, slightly slower |

**✅ Final Model Selected:** XGBoost (balanced precision and recall with best AUC).

---

## 📈 Insights & Recommendations

### **Key Insights**
- Applicants with **higher education** and **prior experience** have a higher approval rate.  
- **Full-time positions** are significantly more likely to be certified than part-time roles.  
- **Prevailing wage** is a strong predictor — higher wages correspond with higher approval likelihood.  
- Employers with a **larger workforce** and **longer establishment history** have better success rates.  
- **Certain regions of employment** consistently show higher approval trends (e.g., Midwest > South).

---

### **Business Recommendations**
1. Encourage employers to **offer full-time roles** and **competitive wages** to strengthen approval chances.  
2. Prioritize applicants with **relevant experience and education** for high-demand occupations.  
3. Provide **data-driven insights** to help employers prepare stronger, compliant applications.  
4. Integrate the trained model into the **OFLC review pipeline** for faster case triage.  
5. Continuously retrain the model with new data to adapt to changing visa patterns.

---

## 🧩 Tech Stack

- **Language:** Python 🐍  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn  
- **Modeling Techniques:** Decision Trees, Ensemble Models, Boosting (XGBoost, Gradient Boosting)  
- **Environment:** Jupyter Notebook / Google Colab  
- **Version Control:** Git & GitHub  

---

## 🚀 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/EasyVisa-VisaApprovalPrediction.git
