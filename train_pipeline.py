# ==============================================================================
# MLOPS UNIT END PROJECT 1: DATA PREPROCESSING, MODEL TOURNAMENT & MLFLOW
# ==============================================================================

import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def extract_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        if title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title in ['Mme']:
            return 'Mrs'
        elif title in ['Mr', 'Mrs', 'Miss', 'Master']:
            return title
        else:
            return 'Rare'
    return 'Rare'

def load_and_preprocess_data(csv_path="titanic.csv"):
    df = pd.read_csv(csv_path)
    
    # 1. Feature Engineering
    df['Title'] = df['Name'].apply(extract_title)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 2. Imputation
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 3. Feature Selection
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = df[features]
    y = df['Survived']
    
    return X, y, df

def main():
    print("=" * 80)
    print("🏥 MLOPS UNIT END PROJECT 1: PREDICTIVE MODELING TOURNAMENT")
    print("=" * 80)
    
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "titanic.csv")
    
    X, y, df_raw = load_and_preprocess_data(csv_path)
    print(f"📊 Dataset Shape: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"🎯 Target Distribution (Survived/Readmitted): {y.mean()*100:.2f}% positive\\n")
    
    # Preprocessing Pipeline
    num_cols = ['Age', 'Fare', 'FamilySize']
    cat_cols = ['Sex', 'Embarked', 'Title', 'Pclass']
    bin_cols = ['IsAlone']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('bin', 'passthrough', bin_cols)
        ]
    )
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"📈 Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples\\n")
    
    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Patient_Readmission_Prediction_MLOps")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = []
    print("=" * 75)
    print(f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<9} {'F1 Score':<9} {'ROC AUC':<8}")
    print("=" * 75)
    
    for name, clf in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob)
            
            mlflow.log_param("model_name", name)
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": roc})
            mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")
            
            print(f"{name:<22} {acc:<10.4f} {prec:<10.4f} {rec:<9.4f} {f1:<9.4f} {roc:<8.4f}")
            results.append({"name": name, "f1": f1, "roc": roc, "pipeline": pipeline})
            
    print("=" * 75)
    
    # Select Best Model based on F1-Score
    results.sort(key=lambda x: x["f1"], reverse=True)
    champion = results[0]
    best_pipeline = champion["pipeline"]
    print(f"\\n🏆 CHAMPION MODEL: {champion['name']} (F1-Score: {champion['f1']:.4f}, ROC-AUC: {champion['roc']:.4f})")
    
    # Save local model artifact for FastAPI & Docker
    model_path = os.path.join(base_dir, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"💾 Saved Champion model pipeline to: {model_path}")
    
    # Save reference test data and current production simulation data for Task 6
    ref_df = X_test.copy()
    ref_df['Actual'] = y_test.values
    ref_df['Prediction'] = best_pipeline.predict(X_test)
    ref_df['Probability'] = best_pipeline.predict_proba(X_test)[:, 1]
    ref_path = os.path.join(base_dir, "reference_data.csv")
    ref_df.to_csv(ref_path, index=False)
    print(f"📂 Saved Baseline Reference Data: {ref_path}")
    
    # Create simulated production dataset with minor shift in Age and Fare
    curr_df = X_test.copy()
    curr_df['Age'] = curr_df['Age'] * np.random.uniform(1.05, 1.15, size=len(curr_df))
    curr_df['Fare'] = curr_df['Fare'] * np.random.uniform(0.95, 1.20, size=len(curr_df))
    curr_df['Actual'] = y_test.values
    curr_df['Prediction'] = best_pipeline.predict(curr_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']])
    curr_df['Probability'] = best_pipeline.predict_proba(curr_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']])[:, 1]
    curr_path = os.path.join(base_dir, "current_production_data.csv")
    curr_df.to_csv(curr_path, index=False)
    print(f"📂 Saved Simulated Current Production Data: {curr_path}")
    
    # Register Champion in MLflow Model Registry
    best_run = mlflow.search_runs(
        experiment_names=["Patient_Readmission_Prediction_MLOps"],
        order_by=["metrics.f1_score DESC"]
    ).iloc[0]
    
    model_uri = f"runs:/{best_run.run_id}/model"
    reg_model = mlflow.register_model(model_uri, "PatientReadmissionChampion")
    print(f"✅ Registered Model in MLflow Model Registry: '{reg_model.name}' (Version {reg_model.version})")
    print("=" * 80)

if __name__ == "__main__":
    main()
