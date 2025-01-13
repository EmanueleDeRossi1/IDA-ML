import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from preprocessing import preprocess
df_not_processed = pd.read_csv("data/train.csv")

df = preprocess(df_not_processed, "replace with median")

# Split data
X = df.drop('stroke', axis=1)
y = df['stroke']

# Divide the data into train and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=133)

# Combine the training data back for upsampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate the minority (stroke == 1) and majority (stroke == 0) classes
stroke_data = train_data[train_data["stroke"] == 1]
no_stroke_data = train_data[train_data["stroke"] == 0]

# Upsample the minority class (stroke == 1)
stroke_data_upsampled = resample(stroke_data, replace=True, n_samples=len(no_stroke_data), random_state=133)

# Combine the upsampled minority class with the majority class
balanced_data_train = pd.concat([no_stroke_data, stroke_data_upsampled])

# Separate the balanced training data into features (X_train_balanced) and target (y_train_balanced)
X_train_balanced = balanced_data_train.drop('stroke', axis=1)
y_train_balanced = balanced_data_train['stroke']

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=133)

# Define the hyperparameter grid
param_grid = {
    # n_estimators: the number of decision trees
    'n_estimators': [20, 50, 100, 200, 300, 500, 1000], 
    # max_depth: the maximum depth each DT can achieve. None means there is no restriction
    'max_depth': [None, 10, 20, 30, 40],
    # min_samples_split: the minimum number of samples required to split a node
    'min_samples_split': [3, 5, 10, 15, 20],
    # the minimum number of samples required to be in a leaf node
    'min_samples_leaf': [3, 5, 10, 15, 20]
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit GridSearchCV on the training data
grid_search.fit(X_train_balanced, y_train_balanced)

# Access the best hyperparameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the eval set
y_pred_prob = best_model.predict_proba(X_eval)[:, 1]
auc_roc = roc_auc_score(y_eval, y_pred_prob)
print("Best Hyperparameters:", best_params)
print(f"AUC ROC Score: {auc_roc:.4f}")

def plot_ROC_curve(y_eval, y_pred_prob):
    # plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_eval, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC ROC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
def plot_feature_importance(best_model):

    feature_importances = best_model.feature_importances_
    feature_names = X_train_balanced.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot the feature importance graph
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout for better readability
    plt.show()

print(plot_feature_importance(best_model))

print(plot_ROC_curve(y_eval, y_pred_prob))