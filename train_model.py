import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_and_save_model():
    # 1. Load dataset
    print("Fetching data...")
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    # 2. Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Preprocessing: Imputation + Scaling
    numeric_features = X.columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 4. Combine using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    # 5. Build pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', KNeighborsRegressor())
    ])

    # 6. Hyperparameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2] # 1: Manhattan distance, 2: Euclidean distance
    }

    # 7. GridSearchCV (5-fold CV)
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )

    # 8. Fit
    grid_search.fit(X_train, y_train)

    # 9. Evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n--- Model Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # 10. Save the pipeline
    with open('california_knn_pipeline.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\n📦 Model saved to 'california_knn_pipeline.pkl'")

if __name__ == "__main__":
    train_and_save_model()