import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

def class_distribution(df):
    """
    Print the percentage of the dataset with class 0 (non-fraud) and class 1 (fraud).
    """
    fraud = round(len(df.loc[df['Class'] == 1]) / len(df) * 100, 2)
    nonfraud = round(len(df.loc[df['Class'] == 0]) / len(df) * 100, 2)
    print('% dataset with class 0: {}%\n% dataset with class 1: {}%\n'.format(nonfraud, fraud))

def get_splits_from_dataframe(df, label='Class', test_size=0.2, random_state=42):
    """
    First separates features and label.Then splits the DataFrame into training and testing sets. 
    """
    X = df.drop(label, axis=1)
    y = df[label]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_irq_from_dataframe(df):
    """
    Calculate the Interquartile Range (IQR) and quartiles for a DataFrame.

    Parameters:
    - df: The DataFrame for which the IQR and quartiles will be calculated.

    Returns:
    - The IQR, lower quartile (Q1), and upper quartile (Q3) as a tuple.
    """
    lower_quartile = df.quantile(0.25)
    upper_quartile = df.quantile(0.75)
    return upper_quartile - lower_quartile, lower_quartile, upper_quartile

def trim_outliers(df, irq, lower_quartile, upper_quartile, slice, threshold=1.5):
    """
    Trim outliers from a DataFrame based on the IQR (Interquartile Range) method.

    Parameters:
    - df: The DataFrame from which outliers will be removed.
    - irq: The Interquartile Range calculated for the data.
    - lower_quartile: The lower quartile (Q1) of the data.
    - upper_quartile: The upper quartile (Q3) of the data.
    - slice: A specific feature slice of the DataFrame.
    - threshold: The outlier threshold (default: 1.5 times the IQR).

    Returns:
    - The DataFrame with outliers removed, lower bound, and upper bound as a tuple.
    """
    lower_bound = lower_quartile - threshold * irq
    upper_bound = upper_quartile + threshold * irq

    outlier_indices = slice[(slice > upper_bound) | (slice < lower_bound)].index
    
    df = df.drop(outlier_indices)

    return df, lower_bound, upper_bound
    
def trim_feature_outliers(df, feats, threshold=1.5, summary=True):
    """
    Trim outliers from specific features of a DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - feats: A list of feature names to trim outliers from.
    - threshold: The outlier threshold (default: 1.5 times the IQR).
    - summary: Whether to print summary information (default: True).

    Returns:
    - The DataFrame with outliers removed.
    """
    trimmed_df = df.copy()

    for feat in feats:
        feat_fraud = trimmed_df[feat].loc[trimmed_df['Class'] == 1]
        irq, lower_quartile, upper_quartile = get_irq_from_dataframe(feat_fraud)
        
        if summary:
            print(f'FEATURE {feat}:')
            print('Dataset size (pre-trim): ', len(trimmed_df))
            print('\nInterquartile Range: {}\nQuartiles: {}\n'.format(irq, [lower_quartile, upper_quartile]))
            
        trimmed_df, lower_bound, upper_bound = trim_outliers(trimmed_df, irq, lower_quartile, upper_quartile, slice=feat_fraud, threshold=threshold)
        
        if summary:
            print('Bounds: {}\n'.format([lower_bound, upper_bound]))
            print('Dataset size (post-trim): ', len(trimmed_df))
            print('---'*45)
    
    return trimmed_df

def get_high_corr_feats(df, min_val, excluded_feats=['Class', 'Class Name', 'Class Dist'], summary=True):
    """
    Find features strongly correlated with the 'Class' column in a DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - min_val: The minimum correlation value for a feature to be considered strongly correlated.
    - excluded_feats: A list of feature names to exclude from consideration (default: ['Class', 'Class Name', 'Class Dist']).
    - summary: Whether to print summary information (default: True).

    Returns:
    - A pandas Series with feature names as the index and their correlations with 'Class' as values.
    """
    class_corr = abs(df.corr()['Class'])
    # don't include class for this (1 correlation with itself)
    corr_feats = class_corr.loc[(class_corr >= min_val) & (~class_corr.index.isin(excluded_feats))]
    
    if summary:
        print('Features Strongly Correlated with Class')
        for idx in corr_feats.index:
            print(idx, corr_feats[idx])

    return corr_feats

def evaluate_model(model, X, y, name='XGBoost', show_cm=True):
    """
    Evaluate a machine learning model and display evaluation metrics.

    Parameters:
    - model: The trained machine learning model.
    - X: The input features for evaluation.
    - y: The true target labels for evaluation.
    - label: A label for the model (default: 'XGBoost').
    - show_cm: Whether to display the confusion matrix plot (default: True).

    Returns:
    - A DataFrame containing evaluation metrics (Accuracy, Precision, Recall, AUC-ROC).
    - The confusion matrix if show_cm is True.
    """
    y_pred = model.predict(X)
    
    acc, precision, recall, roc = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred)

    scores = [acc, precision, recall, roc]
    scores = [round(score, 4) for score in scores]
    scores_df = pd.DataFrame([scores], columns=['Accuracy', 'Precision', 'Recall', 'AUC-ROC'])

    if show_cm:
        _confusion_matrix = plot_confusion_matrix(y, y_pred, name=name)
    else:
        _confusion_matrix = confusion_matrix(y, y_pred)
    
    return scores_df, _confusion_matrix

def train_and_evaluate_model(model, data, val_data=None, label='Class', name='XGBoost', show_cm=True):
    """
    Train and evaluate a machine learning model.

    Parameters:
    - model: The machine learning model to be trained and evaluated.
    - data: The training data DataFrame.
    - val_data: Validation data DataFrame (default: None).
    - label: A label for the model (default: 'XGBoost').
    - show_cm: Whether to display the confusion matrix plot (default: True).

    Returns:
    - A DataFrame containing evaluation metrics (Accuracy, Precision, Recall, AUC-ROC).
    - The confusion matrix if show_cm is True.
    """
    if val_data is None:
        X_train, X_test, y_train, y_test = get_splits_from_dataframe(data)
    else:
        X_test = val_data.drop(label, axis=1)
        y_test = val_data['Class']
        
        # select proper columns from the dataset (data) 
        # and find samples that exclude the testset (val_data)
        mask = ~data.isin(val_data).all(axis=1)
        X_train = data[mask].drop(label, axis=1)
        y_train = data[mask]['Class']
        
    model.fit(X_train, y_train)
    scores, _confusion_matrix = evaluate_model(model, X_test, y_test, name=name, show_cm=show_cm)

    return scores, _confusion_matrix

def plot_confusion_matrix(y_true, y_pred, name='XGBoost'):
    """
    Plot a confusion matrix for evaluating model performance.
    """
    _confusion_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(_confusion_matrix, annot=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    return _confusion_matrix

def plot_roc_curve(model, X, y, name='XGBoost'):
    """
    First predicts based on inputs X, y 
    -> calculates the ROC curve 
    -> plots the ROC curve

    Parameters:
    - model: The trained machine learning model.
    - X: The input features for ROC curve calculation.
    - y: The true target labels for ROC curve calculation.

    Plots:
    - The ROC curve.
    """
    color = sns.color_palette('hls', 8)[5]

    plt.figure(figsize=(7, 5))
    plt.title(f'ROC Curve for {name}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([-.05, 1, 0, 1.05])

    y_pred_proba = model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])


    plt.plot(fpr, tpr, label=name, color=color)

    plt.legend()
    plt.show()

