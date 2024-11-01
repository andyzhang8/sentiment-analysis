from load import load_and_preprocess
from model import create_traditional_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import joblib

def load_test_data(filepath):
    columns = ["textID", "text", "sentiment", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"]
    df = pd.read_csv(filepath, usecols=["text", "sentiment"], encoding="ISO-8859-1", names=columns)
    
    # Map sentiment to labels
    sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)
    
    return df["text"], df["sentiment"]

def train_with_cross_validation(filepath, model_type="logistic_regression", n_splits=5):
    X, _, y, _ = load_and_preprocess(filepath)  # Load for cross-validation
    X = X.tolist()

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_reports = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        
        # Split data
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Create and train model
        model = create_traditional_model(model_type=model_type)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_accuracy)
        fold_reports.append(classification_report(y_test, y_pred, output_dict=True))

        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

    # Calculate average metrics across all folds
    average_accuracy = np.mean(fold_accuracies)
    print("\nAverage Accuracy across all folds:", average_accuracy)
    
    # Overall average classification report
    avg_report = {
        "precision": np.mean([report["weighted avg"]["precision"] for report in fold_reports]),
        "recall": np.mean([report["weighted avg"]["recall"] for report in fold_reports]),
        "f1-score": np.mean([report["weighted avg"]["f1-score"] for report in fold_reports])
    }
    
    print("\nAverage Classification Report across all folds:")
    print(f"Precision: {avg_report['precision']:.4f}")
    print(f"Recall: {avg_report['recall']:.4f}")
    print(f"F1-Score: {avg_report['f1-score']:.4f}")

    return model

def evaluate_on_test(model, test_filepath):
    X_test, y_test = load_test_data(test_filepath)
    y_pred = model.predict(X_test)
    
    # Evaluate model on test data
    print("\nFinal Evaluation on Test Set:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_filepath = "./dataset/train.csv"
    test_filepath = "./dataset/test.csv"
    model_type = "logistic_regression" 
    
    final_model = train_with_cross_validation(train_filepath, model_type=model_type, n_splits=5)
    
    evaluate_on_test(final_model, test_filepath)
    
    model_filename = "sentiment_model.joblib"
    joblib.dump(final_model, model_filename)
    print(f"\nModel saved to {model_filename}")
