# 🏏 T20 World Cup AI Predictor

> An ML-powered Streamlit app that predicts T20 match outcomes using 
> an ensemble model with head-to-head and historical win-rate features.

## 🚀 Live Demo
[Click here to try the app](http://localhost:8507/))  <!-- deploy on Streamlit Cloud (free) -->

## 📸 Screenshots
<img width="1888" height="899" alt="image" src="https://github.com/user-attachments/assets/1480e9a4-0bdc-4c5f-860b-ad896291722a" />
<img width="1043" height="788" alt="image" src="https://github.com/user-attachments/assets/c4251531-9129-427b-9568-97f5cbbdb7dd" />
<img width="1368" height="718" alt="image" src="https://github.com/user-attachments/assets/48577cee-f32b-4f30-9555-4c999baa131c" />
<img width="1429" height="783" alt="image" src="https://github.com/user-attachments/assets/457ed034-7924-4cf8-9eb7-456325bf24d7" />



## 🧠 ML Architecture
- Ensemble: RandomForest + GradientBoosting + Logistic Regression
- Calibrated probabilities via CalibratedClassifierCV
- Features: Team win rate, head-to-head record, label encoding
- 5-Fold Cross-Validation accuracy tracking

## 📦 Tech Stack
Python · Streamlit · Scikit-learn · Pandas · Seaborn · Matplotlib

## 🔧 Setup
pip install -r requirements.txt
 streamlit run  predict_t20_26.py
```
```
requirements
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
