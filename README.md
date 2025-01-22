# Bati Bank Credit Scoring Model

## **Overview**
This project focuses on developing a credit scoring model for Bati Bank to enable a "buy-now-pay-later" service in partnership with an eCommerce platform. The model aims to assess customer creditworthiness and predict loan eligibility based on transaction and user data.

## **Business Objectives**
1. **Categorize Users**: Define proxy variables to label users as high-risk (bad) or low-risk (good).
2. **Predict Credit Risk**: Build models to estimate risk probability for new customers.
3. **Credit Scoring**: Develop a scoring system based on risk probabilities.
4. **Loan Recommendations**: Predict optimal loan amounts and durations for customers.

---

## **Key Deliverables**
### **Data Understanding and EDA**
- **Data Structure**: Explore the structure, rows, columns, and data types.
- **Statistics**: Generate summary statistics for key features.
- **Visualizations**: Analyze distributions and correlations.
- **Missing Values**: Identify and address missing data.
- **Outlier Detection**: Use box plots to spot anomalies.

### **Feature Engineering**
- Aggregate features: Total transaction amounts, averages, and counts per user.
- Extract features: Transaction time, day, and month.
- Encode categorical variables using one-hot and label encoding.
- Handle missing values with imputation or removal.
- Normalize and standardize numerical features.

### **Model Development**
- Train and test multiple models:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Gradient Boosting Machines (GBM)
- Perform hyperparameter tuning with Grid Search and Random Search.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.

### **API Deployment**
- Create a REST API for real-time model predictions using Flask or FastAPI.
- Implement API endpoints to process input data and return predictions.
- Deploy the API to a web server or cloud platform.

---

## **Learning Outcomes**
### **Technical Skills**
- Advanced scikit-learn usage for modeling.
- Feature engineering techniques, including Weight of Evidence (WoE).
- Hyperparameter tuning and model comparison.
- MLOps tools for deployment and monitoring.

### **Knowledge Areas**
- Business reasoning for credit scoring and risk assessment.
- Data exploration and visualization techniques.
- Machine learning and model evaluation.
- Communication of technical results to stakeholders.

---

## **Tools and Technologies**
- **Programming**: Python (Pandas, NumPy, Matplotlib, Scikit-learn, Xverse, WOE).
- **MLOps**: CI/CD pipelines, MLFlow.
- **Data**: Kaggle Xente Challenge dataset.
- **Deployment**: Flask/FastAPI for serving models.

---

## **Folder Structure**
```plaintext
Bati-Bank-Credit-Scoring/
├── data/
│   ├── raw/
│   ├── processed/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_serving.py
├── notebooks/
│   ├── eda.ipynb
│   ├── modeling.ipynb
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
├── requirements.txt
├── README.md
└── LICENSE
```
## **Usage Guide**
### Prerequisites
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   pip install -r requirements.txt
   ```
   ## **Author**
**Yonatan Abrham**  
- Email: [email2yonatan@gmail.com](mailto:email2yonatan@gmail.com)  
- LinkedIn: [Yonatan Abrham](https://www.linkedin.com/in/yonatan-abrham1/)  
- GitHub: [YonInsights](https://github.com/YonInsights)  
Feel free to connect for collaborations or queries.

---

## **Acknowledgements**
- Heartfelt thanks to 10 Academy for providing an excellent internship opportunity.
- Appreciation for the open-source tools and the community that made this project possible.

---

> **"Data is the lifeblood of decision-making, and this project brings it to life."**
