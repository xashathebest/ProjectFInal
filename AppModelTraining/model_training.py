import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import accuracy_score, classification_report,  roc_auc_score # type: ignore
from imblearn.pipeline import Pipeline # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from xgboost import XGBClassifier # type: ignore
import pickle
import os

def load_and_preprocess_data():
    # Load dataset with ALL features
    df = pd.read_csv('diabetes.csv')
    
    # Handle missing values (0s represent missing data in this dataset)
    clinical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in clinical_features:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))
    
    # Drop remaining missing values
    df = df.dropna()

    # Create safe clinical features with epsilon for division
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1e-6)
    df['Insulin_Resistance'] = (df['Glucose'] * df['Insulin']) / 405
    df['Metabolic_Age'] = df['BMI'] * df['Age'] / 10
    df['Diabetes_Risk_Score'] = (df['Glucose']/100) + (df['BMI']/30) + (df['DiabetesPedigreeFunction']*10)

    # Final cleanup of any potential infinite/nan values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    return df.drop('Outcome', axis=1), df['Outcome']

def build_optimized_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(sampling_strategy=0.8, random_state=42)),
        ('xgb', XGBClassifier( # type: ignore
            n_estimators=500,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=3,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        ))
    ])

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build and train model
    model = build_optimized_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.Series(model.named_steps['xgb'].feature_importances_, index=X.columns)
    print("\nTop 10 Features:")
    print(feature_importance.sort_values(ascending=False).head(10))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
