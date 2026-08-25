# Sleep Disorder Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)]()

An interactive machine learning web application that evaluates a patient's lifestyle and medical details to predict Insomnia or Sleep Apnea[cite: 14]. 

---

## Features

* **Multi-Factor Input Pipeline:** Evaluates user inputs organized into four categories: Demographics & Work, Lifestyle Habits, Medical Vitals, and Sleep Metrics & Environment[cite: 14].
* **K-Nearest Neighbors Classification:** Utilizes a trained `KNeighborsClassifier` to predict if a patient has Insomnia, Sleep Apnea, or no disorder (None)[cite: 12, 14, 16].
* **Dynamic Health Recommendations:** Evaluates specific risk factors, such as caffeine intake over 200mg, stress levels above 7, high blood pressure, and noise levels over 50dB, to generate tailored medical and lifestyle advice[cite: 14].
* **Robust Error Handling:** Safely halts execution and displays error messages if required `.pkl` files are missing or if the user inputs unseen categorical values[cite: 14].

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Machine Learning Core** | `scikit-learn`[cite: 12, 16] |
| **Data Processing** | `pandas` / `numpy`[cite: 14] |
| **Frontend / UI** | `streamlit`[cite: 14] |
| **Model Serialization** | `joblib`[cite: 14] |

---

## Project Structure

```text
├── Run_This.py             # Main Streamlit application and inference logic[cite: 14]
├── knn_model.pkl           # Trained KNeighborsClassifier model[cite: 12]
├── scaler.pkl              # StandardScaler for feature scaling[cite: 15]
├── label_encoders.pkl      # Encoders for categorical features like Occupation and BMI[cite: 13, 14]
├── target_encoder.pkl      # Encoder for the target labels (Insomnia, Sleep Apnea, None)[cite: 14, 16]
├── .gitignore              # Git ignore file[cite: 11]
└── README.md
