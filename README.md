# **🌱 Crop & Fertilizer Recommendation System**  

![Project Banner](https://your-banner-image-url.com) *(Optional: Add a banner image related to agriculture & AI)*  

## **📌 Introduction**  
This project is a **Machine Learning-based Crop & Fertilizer Recommendation System**, developed as part of my **AICTE-Shell Internship**. The system provides **personalized crop and fertilizer recommendations** based on soil composition, weather conditions, and crop requirements.  

🔗 **Deployed Application:** [Crop & Fertilizer Recommendation System](https://samarth4023-aicte-shell-internship-rs-app-nhp1xt.streamlit.app/)  

## **🎯 Project Goals**  
✅ Integrate **AI and Green Skills** into agriculture.  
✅ Explore **Machine Learning techniques** for precision farming.  
✅ Develop a **real-time recommendation system** for farmers.  
✅ Enhance **data preprocessing, model selection, and optimization skills**.  
✅ Deploy an **interactive web app** using **Streamlit**.  

## **📂 Datasets Used**  
The project uses two datasets:  

### **1️⃣ Crop Dataset**  
- **Features:** Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall  
- **Target Variable:** Crop Label (Crop Name)  

### **2️⃣ Fertilizer Dataset**  
- **Features:** Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Phosphorous, Potassium  
- **Target Variable:** Fertilizer Name  

## **📊 Methodology**  

### **Step 1: Data Collection & Preprocessing**  
🔹 **Performed Exploratory Data Analysis (EDA)** to understand data distributions.  
🔹 **Encoded categorical variables** using Label Encoding.  
🔹 **Split the dataset** into **train & test sets** using Scikit-Learn.  

### **Step 2: Model Selection & Training**  
🔹 Tested multiple **ML algorithms**, including:  
   - Logistic Regression, GaussianNB, SVC, KNN, DecisionTree, ExtraTree, RandomForest, Bagging, Gradient Boosting, AdaBoost, CatBoost, and LGBM.  
🔹 Compared **model performance** based on accuracy & validation metrics.  

### **Step 3: Ensemble Learning for Optimization**  
🔹 Evaluated different ensemble techniques:  
   - **Voting Classifier, Stacking, Averaging Probabilities, Weighted Ensemble, Blend Ensemble (Custom Blending).**  
🔹 **Blend Ensemble provided the best results**, so it was selected.  
🔹 **Cross-validation** was performed to validate model performance.  

### **Step 4: Deployment & Integration**  
🔹 Exported trained models as **.pkl files** (`crop_recommendation.pkl`, `fertilizer_recommendation.pkl`).  
🔹 Integrated models into a **Streamlit app** for real-time predictions.  
🔹 **Deployed the application on Streamlit Cloud.**  

## **🔍 Key Features**  
✅ **Real-time Model Performance Metrics** (Accuracy Tracking).  
✅ **Feature Importance Analysis** for better interpretability.  
✅ **Feature Distributions** to understand data variations.  
✅ **Prediction Probabilities** to assess model confidence.  

## **🚀 Technologies Used**  
| Category            | Tools & Libraries |
|---------------------|------------------|
| **Development**    | Python, Jupyter Notebook, Anaconda |
| **ML Frameworks**  | Scikit-Learn, CatBoost, LGBM, XGBoost |
| **Data Processing**| Pandas, NumPy |
| **Visualization**  | Matplotlib, Seaborn |
| **Deployment**     | Streamlit, Streamlit Cloud |

## **📷 Screenshots**  

| **Crop Recommendation Model** | **Fertilizer Recommendation Model** |
|------------------------------|--------------------------------|
| ![Crop Model](https://your-image-url.com) | ![Fertilizer Model](https://your-image-url.com) |

| **Streamlit App - Crop Recommendation** | **Streamlit App - Fertilizer Recommendation** |
|-----------------------------------------|-----------------------------------------------|
| ![App Screenshot](https://your-image-url.com) | ![App Screenshot](https://your-image-url.com) |

## **🎯 Future Improvements**  
🔹 Expand the dataset to include **more crop varieties & soil types**.  
🔹 Integrate **real-time weather data** for better recommendations.  
🔹 Incorporate **IoT & satellite data** for advanced precision farming.  
🔹 Optimize **model efficiency & deployment** for faster predictions.  

## **📥 Installation & Setup**  

### **🔹 Clone the Repository**  
```bash
git clone https://github.com/Samarth4023/Shell-Internship.git
cd Shell-Internship
```

### **🔹 Install Required Dependencies**  
```bash
pip install -r requirements.txt
```

### **🔹 Run the Streamlit App**  
```bash
streamlit run app.py
```

## **📜 License**  
This project is **open-source** and free to use. Feel free to contribute!  

## **📧 Contact**  
📌 **Author:** Samarth Pujari  
📌 **GitHub:** [Samarth4023](https://github.com/Samarth4023)  
📌 **LinkedIn:** [Your LinkedIn Profile]  

---

This **README.md** is now structured, detailed, and **ready for GitHub**! Let me know if you'd like any refinements! 🚀😊
