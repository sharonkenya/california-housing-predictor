California Housing Price Predictor 🏠

An end-to-end machine learning workflow to predict median house values in California using K-Nearest Neighbors (KNN) Regression.

📝 Project Overview

This project demonstrates a complete machine learning lifecycle, from data acquisition and preprocessing to model tuning and deployment. Using the California Housing dataset, I built a pipeline that predicts house prices (in $100,000s) based on census data such as median income, house age, and geographic location.

🚀 Features

Automated Pipeline: Combines data imputation and feature scaling using ColumnTransformer.
Hyperparameter Tuning: Utilizes GridSearchCV with 5-fold cross-validation to find the optimal KNN configuration.

Model Persistence: Saves the trained model as a .pkl file for production use.
REST API Deployment: A functional Flask API with a /predict endpoint to serve real-time predictions.

🛠️ Technologies Used

Python (v3.x)  
Scikit-learn: For modeling, preprocessing, and evaluation.  
Pandas: For data manipulation.  
Flask: For creating the web API.  
Pickle: For model serialization.

📁 Project Structure

code  
Text  
├── train_model.py # Script to train and save the model
├── app.py # Flask API for model deployment
├── california_knn_pipeline.pkl # The saved trained model (generated after training)
└── README.md # Project documentation

⚙️ Setup & Installation

Clone the repository:  
code  
Bash  
git clone https://github.com/your-username/california-housing-predictor.git  
cd california-housing-predictor  
Install dependencies:  
code  
Bash  
pip install pandas scikit-learn flask

🏃 How to Run

1. Train the Model

   Run the training script to preprocess the data, perform the grid search, and save the best model:  
   code  
   Bash  
   python train_model.py  
   This will output the best R² score and save california_knn_pipeline.pkl.

2. Start the API

   Run the Flask application to start the local server:  
   code  
   Bash  
   python app.py  
   The server will start at http://127.0.0.1:5000.

3. Test the Prediction

   You can send a POST request to the /predict endpoint using a tool like Postman or curl:
   code  
   Bash  
   curl -X POST http://127.0.0.1:5000/predict \
    -H "Content-Type: application/json" \
    -d '[{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.0, "Population": 322, "AveOccup": 2.5, "Latitude": 37.8, "Longitude": -122.2}]'

   📊 Model Evaluation

   The model was tuned using GridSearchCV across the following parameters:  
   n_neighbors: [3, 5, 7, 9]  
   weights: ['uniform', 'distance']  
   p: [1, 2]

   Performance Metric: R² (Coefficient of Determination) was used to ensure the model accurately captures the variance in housing prices.

   🎓 Conclusion

   Through this project, I gained hands-on experience in:  
   Building robust ML pipelines to prevent data leakage.  
   Optimizing distance-based algorithms like KNN.  
   Deploying models as functional web services.
