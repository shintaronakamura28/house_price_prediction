import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
#metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression_model(model, X_train_transformed_df, y_train, X_test_transformed_df, y_test):
    # Create predictions
    train_preds = model.predict(X_train_transformed_df)
    test_preds = model.predict(X_test_transformed_df)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    # Print metrics
    print("Training set metrics:")
    print(f"MAE: {train_mae}")
    print(f"MSE: {train_mse}")
    print(f"R²: {train_r2}")
    print()
    print("Testing set metrics:")
    print(f"MAE: {test_mae}")
    print(f"MSE: {test_mse}")
    print(f"R²: {test_r2}")

    # Plot predictions vs true values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_train, train_preds, alpha=0.3)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', linewidth=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title('Train Set: True vs Predicted Values')
    
    axes[1].scatter(y_test, test_preds, alpha=0.3)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predictions')
    axes[1].set_title('Test Set: True vs Predicted Values')
    
    plt.show()

    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
    }

if __name__ == '__main__':
    pass

def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f'Model Saved to {model_path}')