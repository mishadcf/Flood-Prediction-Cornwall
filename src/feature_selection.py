import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


def compute_correlation_matrix(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix for the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target.
    - target (str): Name of the target column.
    
    Returns:
    - pd.DataFrame: Correlation matrix sorted by correlation with the target.
    """
    corr_matrix = df.corr()
    corr_with_target = corr_matrix[target].abs().sort_values(ascending=False)
    return corr_matrix, corr_with_target


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str = None):
    """
    Plots a heatmap of the correlation matrix.
    
    Parameters:
    - corr_matrix (pd.DataFrame): Correlation matrix.
    - output_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_feature_importances(model, feature_names: list, top_n: int = 20, output_path: str = None):
    """
    Plots the feature importances from a model.
    
    Parameters:
    - model: Trained model with feature_importances_ attribute.
    - feature_names (list): List of feature names.
    - top_n (int): Number of top features to display.
    - output_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def train_random_forest(df: pd.DataFrame, target: str, random_state: int = 42):
    """
    Trains a Random Forest Regressor and returns the trained model and feature names.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target.
    - target (str): Name of the target column.
    - random_state (int): Seed for reproducibility.
    
    Returns:
    - model: Trained Random Forest model.
    - feature_names (list): List of feature names used for training.
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=random_state
    )
    
    # Initialize and train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Random Forest Validation RMSE: {rmse:.4f}")
    print(f"Random Forest Validation MAE: {mae:.4f}")
    
    return rf, X.columns.tolist()


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Variance Inflation Factor (VIF) for each feature in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features.
    
    Returns:
    - pd.DataFrame: DataFrame with features and their corresponding VIF.
    """
    vif_data = pd.DataFrame()
    vif_data['feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data


def main_feature_selection(df: pd.DataFrame, target: str, output_dir: str = '../reports/figures/'):
    """
    Main function to perform feature correlation analysis and feature importance evaluation.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target.
    - target (str): Name of the target column.
    - output_dir (str): Directory to save the plots.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Correlation Analysis
    corr_matrix, corr_with_target = compute_correlation_matrix(df, target)
    
    # Save correlation matrix to a CSV
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    
    # Plot correlation heatmap
    heatmap_path = os.path.join(output_dir, 'feature_correlation_heatmap.png')
    plot_correlation_heatmap(corr_matrix, output_path=heatmap_path)
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    # Display top correlations with target
    print("\nTop correlations with target:")
    print(corr_with_target.head(20))
    
    # 2. Feature Importance using Random Forest
    rf_model, feature_names = train_random_forest(df, target)
    
    # Plot feature importances
    feature_importance_path = os.path.join(output_dir, 'random_forest_feature_importances.png')
    plot_feature_importances(rf_model, feature_names, top_n=20, output_path=feature_importance_path)
    print(f"Feature importances plot saved to {feature_importance_path}")
    
    # 3. Optional: Save feature importances to CSV
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    feature_importance_df.to_csv(os.path.join(output_dir, 'random_forest_feature_importances.csv'), index=False)
    print(f"Feature importances saved to {os.path.join(output_dir, 'random_forest_feature_importances.csv')}")
    
    # 4. Additional Analysis: Pairplot for Top Features (Optional)
    top_n = 10
    top_features = feature_importance_df['feature'].head(top_n).tolist()
    sns.pairplot(df[top_features + [target]], diag_kind='kde')
    pairplot_path = os.path.join(output_dir, 'pairplot_top_features.png')
    plt.savefig(pairplot_path)
    plt.close()
    print(f"Pairplot of top {top_n} features saved to {pairplot_path}")
    
    print("\nFeature selection and analysis completed.")


if __name__ == "__main__":
    import pandas as pd
    # from data_preprocessing import preprocess_data  # Assuming you have this function
    
    # Define paths
    raw_data_path = '../data/raw/woolstone_river_weather.csv'
    
    # # Preprocess data and generate features
    # df_features = preprocess_data(raw_data_path)
    
    # Define target column
    target_column = 'target_t_plus_6'
    
    # Perform feature selection analysis
    main_feature_selection(df_features, target=target_column)
