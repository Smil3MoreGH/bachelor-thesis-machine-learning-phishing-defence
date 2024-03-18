import matplotlib.pyplot as plt  # Used for creating static, interactive, and animated visualizations
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Provides high-performance, easy-to-use data structures and data analysis tools
import re  # Provides regular expression matching operations
import nltk  # A leading platform for building Python programs to work with human language data
import time  # Provides various time-related functions
import warnings  # Used to warn the developer of situations that aren’t necessarily exceptions

from imblearn.over_sampling import SMOTE  # Implements SMOTE - Synthetic Minority Over-sampling Technique
from joblib import dump, load  # Provides tools for saving and loading Python objects that make use of NumPy data structures

import tensorflow as tf  # An end-to-end open-source platform for machine learning
from tensorflow.keras.models import Sequential  # Allows creation of a linear stack of layers in the model
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D  # Provides layers commonly used for deep learning models
from tensorflow.keras.optimizers import Adam  # Implements the Adam optimization algorithm
from tensorflow.keras.callbacks import EarlyStopping  # Allows training to stop early based on the performance of the validation set
from tensorflow.keras.utils import to_categorical  # Converts a class vector (integers) to binary class matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Used for sequence padding
from tensorflow.keras.preprocessing.text import Tokenizer  # Text tokenization utility class

from nltk.corpus import stopwords  # A list of common words that are usually ignored in text processing
from nltk.stem import WordNetLemmatizer  # Used for lemmatizing words to their root form

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer  # Transform text to vector based on the frequency of each word that occurs in the entire text
from sklearn.linear_model import LogisticRegression  # Implements logistic regression for binary classification tasks
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve  # Provides tools to evaluate the model performance
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay  # More tools for model evaluation, specifically for binary classification
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split  # Tools for splitting data, cross-validation, and model evaluation
from sklearn.naive_bayes import MultinomialNB  # Implements the naive Bayes algorithm for multinomially distributed data
from sklearn.preprocessing import LabelEncoder  # Used to encode target labels with value between 0 and n_classes-1

from wordcloud import WordCloud  # Used to generate a word cloud image from text

warnings.filterwarnings('ignore')  # Suppresses warnings

# Downloading necessary NLTK data sets for text processing
nltk_data_needed = ['wordnet', 'stopwords']
for dataset in nltk_data_needed:
    if not nltk.download(dataset, quiet=True):
        nltk.download(dataset)

print("Loading dataset...")
df = pd.read_csv("Phishing_Email.csv")
print("Dataset successfully loaded. Displaying the first few entries:")
print(df.head(), '\n')

# Defining functions to count URLs and check for HTML content in emails
def count_urls(text):
    text = str(text)  # Ensuring the text is a string
    urls_http = text.count('http')
    return urls_http

def contains_html(text):
    text = str(text)  # Ensure text is in string format
    html_pattern = re.compile('<.*?>')
    if re.search(html_pattern, text):
        return 1
    else:
        return 0

# Adding new features to the dataset based on the email content
df['URLs Count'] = df['Email Text'].apply(count_urls)
df['Contains HTML'] = df['Email Text'].apply(contains_html)

# Text preprocessing to prepare data for modeling
def preprocess_text(text):
    text = text.lower()
    text = text.strip()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

print("Preprocessing data...")
df = df.drop(['Unnamed: 0'], axis=1).dropna()
df['Email Text'] = df['Email Text'].apply(preprocess_text)
df['Length'] = df['Email Text'].apply(len)
df['Email Type'].replace(['Safe Email', 'Phishing Email'], [0, 1], inplace=True)
print("Data preprocessing finished. Displaying the updated first few entries:")
print(df.head(), '\n')

# Visualizing most common words in phishing emails using a word cloud
print("Generating word cloud for email text visualization...")
all_mails = " ".join(df['Email Text'])
word_cloud = WordCloud(stopwords=set(stopwords.words('english')), width=900, height=700, mode="RGB").generate(all_mails)
plt.figure(figsize=(10, 6))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Visualizing class distribution before applying SMOTE to address class imbalance
print("Analyzing class distribution prior to SMOTE application...")
class_counts = df['Email Type'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['Safe Email', 'Phishing Email'])
plt.ylabel('Count')
plt.title('Email Type Distribution Before SMOTE')
plt.show()


# _________________________Vectorization method selection here_________________________


# Vectorizing email text to prepare for machine learning modeling
vectorization_method = 'tfidf'

y = df['Email Type'].values
if vectorization_method == 'tfidf':
    current_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
elif vectorization_method == 'count':
    current_vectorizer = CountVectorizer(stop_words='english', max_features=20000)
elif vectorization_method == 'hashing':
    print("Applying HashingVectorizer on email text...")
    current_vectorizer = HashingVectorizer(stop_words='english', n_features=2**15)

if vectorization_method in ['tfidf', 'count']:
    X = current_vectorizer.fit_transform(df['Email Text']).toarray()
else:  # For 'hashing'
    X = current_vectorizer.transform(df['Email Text']).toarray()

# Saving the vectorizer for later use
dump(current_vectorizer, 'LR_vectorizer.joblib')

print(f"Text data vectorized using {vectorization_method.upper()}. Observing shapes of X and y:")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape, '\n')

# Addressing class imbalance using SMOTE to enhance model performance
print("Implementing SMOTE for class balance enhancement...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Post-SMOTE class distribution:")
resampled_class_counts = pd.Series(y_resampled).value_counts()
print(resampled_class_counts, '\n')

# Visualizing class distribution after SMOTE application
print("Visual representation of class distribution following SMOTE enhancement...")
plt.figure(figsize=(8, 6))
plt.bar(resampled_class_counts.index, resampled_class_counts.values, color=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['Safe Email', 'Phishing Email'])
plt.ylabel('Count')
plt.title('Post-SMOTE Email Type Distribution')
plt.show()

# Splitting dataset into training and testing parts for model evaluation
print("Dividing dataset into training and testing subsets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Dataset division completed. Shapes of training and testing sets displayed:")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape, '\n')

# Preprocessing and cleaning further by removing outliers based on URL count
filtered_df = df[df['URLs Count'] <= 25]
print("Descriptive statistics for new features after removing emails with >25 URLs:")
print(filtered_df[['URLs Count', 'Contains HTML']].describe(), '\n')


# _________________________Model section starts here_________________________


# Cross-validating and optimizing the logistic regression model for balanced classification
print("Performing cross-validation to gauge baseline model performance...")
model_cv = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
cv_scores = cross_val_score(model_cv, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores obtained:", cv_scores)
print("Computed average accuracy score from cross-validation:", cv_scores.mean(), '\n')

# Timing model training and optimization with GridSearchCV
training_start_time = time.time()
print("Optimizing regularization parameter via GridSearchCV to refine model performance...")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')

training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f"Training completed in {training_time:.3f} seconds")

# Evaluating the optimized model on the test set
model = grid_search.best_estimator_
prediction_start_time = time.time()
print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)
print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

# Displaying the outcomes of GridSearchCV to identify optimal model parameters and cross-validation accuracy
print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')

# Initializing the model with the best parameters found during grid search for future predictions
model = grid_search.best_estimator_

# Timing the prediction process to evaluate the model's efficiency
prediction_start_time = time.time()

# Making predictions on the test dataset using the optimized model to assess its performance
print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)

# Calculating the duration of the prediction process to understand the model's speed in application scenarios
prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time
print(f"Prediction completed in {prediction_time:.3f} seconds")


# _________________________Visualization starts here_________________________


# Analyzing the model's predictive performance through classification metrics and a confusion matrix
print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

# Displaying the confusion matrix in a visual format for better understanding of model performance
print("Generating visual display of the confusion matrix for enhanced interpretability...")
cm = confusion_matrix(y_test, y_pred)  # Calculating the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe Email', 'Phishing Email'])  # Creating display object
disp.plot(cmap='Blues')  # Plotting with a color scheme for clarity
plt.title(f'Confusion Matrix for Model: LR with ({vectorization_method.upper()})')
plt.show()

# Analyzing model performance over various amounts of training data to identify learning trends
print("Generating learning curve to assess model training performance...")
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
train_mean = np.mean(train_scores, axis=1)  # Average training scores
train_std = np.std(train_scores, axis=1)  # Standard deviation of training scores
test_mean = np.mean(test_scores, axis=1)  # Average cross-validation scores
test_std = np.std(test_scores, axis=1)  # Standard deviation of cross-validation scores
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="gray")  # Training score variance
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="gainsboro")  # Test score variance
plt.plot(train_sizes, train_mean, 'o-', color="red", label="Training score")  # Training score line
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")  # Cross-validation score line
plt.title(f"Learning Curve for Model: LR with ({vectorization_method.upper()})")
plt.xlabel("Number of Training Examples")
plt.ylabel("Model Score")
plt.legend(loc="best")
plt.show()

# Evaluating the model's discriminative ability using ROC curve and AUC metrics
print("Calculating ROC curve and AUC to evaluate model's discriminative ability...")
y_score = model.decision_function(X_test)  # Model scores
fpr, tpr, _ = roc_curve(y_test, y_score)  # False positive and true positive rates
roc_auc = auc(fpr, tpr)  # Area under the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # Plotting ROC curve
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Model: LR with ({vectorization_method.upper()})')
plt.legend(loc="lower right")
plt.show()

# Examining the model's feature importance to identify what factors contribute to its predictions
print("Analyzing feature importance for insights into influential factors in phishing email detection...")
top_n_features = 20  # Number of top features to visualize
importance = model.coef_[0]  # Model coefficients as importance indicator
feature_names = np.array(current_vectorizer.get_feature_names_out())  # Feature names for labeling
# Indices of top positive and negative influential features
top_positive_indices = np.argsort(importance)[-top_n_features:]
top_negative_indices = np.argsort(importance)[:top_n_features]
combined_indices = np.hstack([top_negative_indices, top_positive_indices])  # Combine for overall top features
combined_features = feature_names[combined_indices]  # Corresponding feature names
# Visualizing feature importance
colors = ['tomato' if x < 0 else 'dodgerblue' for x in importance[combined_indices]]  # Color code by influence direction
plt.figure(figsize=(12, 10))
plt.barh(range(2 * top_n_features), importance[combined_indices], color=colors)
plt.yticks(range(2 * top_n_features), combined_features)
plt.xlabel('Coefficient Magnitude')
plt.title(f'Top {top_n_features} Influential Features for Model: LR with ({vectorization_method.upper()})')
plt.gca().invert_yaxis()
plt.axvline(x=0, color='grey', linestyle='--')
plt.show()

# Evaluating model performance through the Precision-Recall curve, providing insight into the balance between precision and recall for different thresholds
y_scores = model.predict_proba(X_test)[:, 1]  # Predicted probabilities
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)  # Precision and recall values
display = PrecisionRecallDisplay(precision=precision, recall=recall)
display.plot()
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plotting Cumulative Gain Curve to assess how much better the model is compared to random guessing
def plot_cumulative_gain(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    cum_true_positives = np.cumsum(y_true_sorted)
    cum_gains = cum_true_positives / cum_true_positives[-1]  # Cumulative gains
    baseline = np.linspace(0, 1, len(cum_gains))  # Baseline for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(cum_gains, label='Model')
    plt.plot(baseline, label='Baseline', linestyle='--')
    plt.title('Cumulative Gains Curve')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Cumulative gain')
    plt.legend()
    plt.show()
plot_cumulative_gain(y_test, y_scores)  # Applying function to plot curve


# _________________________Saving model starts here_________________________
# https://www.kaggle.com/datasets/phangud/spamcsv


# Saving the model to disk for future predictions, demonstrating the process of model persistence for deployment
dump(model, 'LR_trained_model.joblib')  # Model saved

# Loading the model for subsequent prediction tasks, simulating a real-world application scenario
model = load('LR_trained_model.joblib')  # Model loaded
vectorizer = load('LR_vectorizer.joblib')  # Vectorizer loaded

# Preparing a new dataset for prediction by the trained model, demonstrating the model's application to unseen data
print("Loading dataset 2...")
df_new = pd.read_csv("Phishing_SMS.csv", delimiter=';')  # New dataset loaded
print("Dataset successfully loaded. Displaying the first few entries:")
print(df_new.head(), '\n')

df_new = df_new.dropna()  # Cleaning dataset

# Applying preprocessing steps to the new dataset to match training data format
df_new['Email Text'] = df_new['Email Text'].apply(preprocess_text)
df_new['URLs Count'] = df_new['Email Text'].apply(count_urls)
df_new['Contains HTML'] = df_new['Email Text'].apply(contains_html)
df_new['Length'] = df_new['Email Text'].apply(len)

X_new = vectorizer.transform(df_new['Email Text']).toarray()  # Vectorizing new data

df_new['Email Type'] = df_new['Email Type'].replace(['Safe Email', 'Phishing Email'], [0, 1])  # Encoding labels
y_new = df_new['Email Type'].values
y_new = y_new.astype(int)
print("Unique values in y_new:", np.unique(y_new))

# Addressing class imbalance in the new dataset with SMOTE, ensuring model training and prediction processes are applicable to balanced data
smote = SMOTE(random_state=42)
X_new_resampled, y_new_resampled = smote.fit_resample(X_new, y_new)

print("After SMOTE:")
print("Unique values in y_new_resampled:", np.unique(y_new_resampled))

# Predicting with the trained model on the new, resampled data and evaluating model performance, demonstrating the model's adaptability and predictive capability on different datasets
y_pred_new_resampled = model.predict(X_new_resampled)
print(classification_report(y_new_resampled, y_pred_new_resampled))
print("Confusion matrix output:")
print(confusion_matrix(y_new_resampled, y_pred_new_resampled))
