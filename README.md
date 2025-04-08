# Predict-the-onset-of-diabetes-based-on-diagnostic

#### Overview

This project aims to build a machine learning-based predictive model to determine the likelihood of a patient developing diabetes based on diagnostic health parameters. It uses the Pima Indians Diabetes dataset and applies various classification algorithms to analyze patterns and produce actionable health insights.

#### Dataset Overview

The dataset is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases. It includes diagnostic measurements from female patients of Pima Indian heritage aged 21 and above.

- **Total Records**: 768
- **Target Variable**: `Outcome` (1 = Diabetes, 0 = No Diabetes)

#### Objectives

- Analyze and understand patterns within the diabetes dataset.
- Handle missing or biologically implausible values in medical features.
- Train multiple machine learning models to classify patients as diabetic or non-diabetic.
- Evaluate and compare model performance.
- Generate business-level insights and strategies based on findings.

#### Dataset Features

| Feature                    | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Pregnancies               | Number of times pregnant                                     |
| Glucose                   | Plasma glucose concentration (mg/dL)                         |
| BloodPressure             | Diastolic blood pressure (mm Hg)                            |
| SkinThickness             | Triceps skin fold thickness (mm)                            |
| Insulin                   | 2-Hour serum insulin (mu U/ml)                              |
| BMI                       | Body mass index (weight in kg/(height in m)^2)              |
| DiabetesPedigreeFunction  | Diabetes likelihood based on family history                 |
| Age                       | Age in years                                                 |
| Outcome                   | Class variable (0 = No diabetes, 1 = Diabetes)              |


#### Business Insights & Strategies

- **Glucose and BMI** are highly correlated with the onset of diabetes and should be prioritized in health screenings.
- **Preventative measures** such as regular blood sugar testing and lifestyle management should be promoted for at-risk individuals (e.g., high BMI, older age).
- **Family history** (as reflected by Diabetes Pedigree Function) is a strong risk factor, highlighting the need for personalized screening strategies.
- Machine learning models can be integrated into early diagnostic tools in clinics to flag high-risk patients in real-time.

#### Tools & Technologies

- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **IDE**: Jupyter Notebook
- **Models Applied**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

#### Key Findings

- **Random Forest** and **SVM** classifiers achieved the highest accuracy on the test set.
- **Glucose**, **BMI**, and **Age** are the most significant predictors of diabetes.
- Replacing medically invalid zero values and scaling features significantly improved model performance.
- Visualizations like pairplots and heatmaps were useful in understanding data relationships.

#### Conclusion

This project demonstrates that machine learning algorithms can be effectively used to predict diabetes from basic health metrics. With further tuning and integration of more diverse patient data, such models can support early intervention strategies and reduce healthcare costs.

