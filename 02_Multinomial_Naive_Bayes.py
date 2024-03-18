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


# Employing cross-validation to assess baseline performance and ensure model robustness across dataset partitions
print("Performing cross-validation to gauge baseline model performance...")
model_cv = MultinomialNB()
cv_scores = cross_val_score(model_cv, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores obtained:", cv_scores)
print("Computed average accuracy score from cross-validation:", cv_scores.mean(), '\n')

# Initiating model training timing to measure computational efficiency and effectiveness
training_start_time = time.time()

# Tuning the 'alpha' parameter of MultinomialNB for optimal smoothing and performance
print("Exploring alpha parameter with GridSearchCV to refine model performance...")
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')

# Calculating the duration of the training process to evaluate time efficiency
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f"Training completed in {training_time:.3f} seconds")

# Configuring the model with the best parameters for prediction readiness and accuracy assessment on new data
model = grid_search.best_estimator_

# Starting prediction timing to assess real-world application speed and efficiency
prediction_start_time = time.time()

# Making predictions to evaluate the model's generalization capability and classification accuracy
print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)
print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

# Calculating prediction process duration to understand application feasibility and speed
prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time
print(f"Prediction completed in {prediction_time:.3f} seconds")

# Reiterating the achievement of optimal model parameters and performance for documentation and review
print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')

# Finalizing model setup with optimized parameters for enhanced prediction accuracy and re-evaluating model outcomes
model = grid_search.best_estimator_
prediction_start_time = time.time()

print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)

print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time
print(f"Prediction completed in {prediction_time:.3f} seconds")


# _________________________Visualization starts here_________________________


# Visualizing the Confusion Matrix
print("Generating visual display of the confusion matrix for enhanced interpretability...")
# Calculate the confusion matrix using true labels and predictions
cm = confusion_matrix(y_test, y_pred)
# Create a display object for the confusion matrix with labels for clarity
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe Email', 'Phishing Email'])
# Plot the confusion matrix with a specific color map
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix for Model: MNB with ({vectorization_method.upper()})')
plt.show()

# Generating and Plotting the Learning Curve
# This helps to understand how the model's performance varies with the amount of training data
print("Generating learning curve to assess model training performance...")
# Calculate training and test scores across different training set sizes
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
# Calculate the mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Visualize the learning curve with shaded areas representing the variance
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="gainsboro")
plt.plot(train_sizes, train_mean, 'o-', color="red", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")
plt.title(f"Learning Curve for Model: MNB with ({vectorization_method.upper()})")
plt.xlabel("Number of Training Examples")
plt.ylabel("Model Score")
plt.legend(loc="best")
plt.show()

# Calculating and Plotting the ROC Curve and AUC
print("Calculating ROC curve and AUC to evaluate model's discriminative ability...")
# Get the probability estimates for each prediction
y_score = model.predict_proba(X_test)[:, 1]
# Calculate False Positive Rate, True Positive Rate, and thresholds
fpr, tpr, _ = roc_curve(y_test, y_score)
# Calculate the Area Under the Curve (AUC) for the ROC
roc_auc = auc(fpr, tpr)
# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Model: MNB with ({vectorization_method.upper()})')
plt.legend(loc="lower right")
plt.show()

# Analyzing Feature Significance based on Log Probabilities for Naive Bayes
print("Analyzing feature significance based on log probabilities for Naive Bayes...")
top_n_features = 20  # Setting the number of top features to visualize for better interpretability

# Accessing the log probability of features given a class
feature_log_prob = model.feature_log_prob_
# Retrieving the names of features from the vectorizer
feature_names = np.array(current_vectorizer.get_feature_names_out())
# Identifying the indices of the top features for each class
top_features_per_class = np.argsort(feature_log_prob, axis=1)[:, -top_n_features:]

import matplotlib.pyplot as plt
import numpy as np
# Assuming 'feature_log_prob' and 'feature_names' are already defined as in your previous context
# Also assuming 'model' is already trained and is a MultinomialNB model instance
top_n_features = 20  # Number of top features to show
# Plot top and bottom indicative features for each class
def plot_feature_importance(class_index, feature_log_prob, feature_names, top_n_features):
    # Sorting indices by feature log probability
    sorted_indices = np.argsort(feature_log_prob[class_index])
    # Combine top bottom indices
    combined_indices = np.concatenate([sorted_indices[:top_n_features], sorted_indices[-top_n_features:]])
    # Get corresponding feature names and log probabilities
    features = feature_names[combined_indices]
    log_probs = feature_log_prob[class_index, combined_indices]

    # Colors for the bars
    colors = ['red' if i < top_n_features else 'blue' for i in range(2 * top_n_features)]

    # Plot
    plt.figure(figsize=(12, 8))
    y_vals = np.arange(2 * top_n_features)
    plt.barh(y_vals, log_probs, color=colors, edgecolor='black')
    plt.yticks(y_vals, features)
    plt.gca().invert_yaxis()  # To display the top feature at the top
    plt.xlabel('Log Probability')
    class_name = 'Phishing Email' if class_index == 1 else 'Safe Email'
    plt.title(f'Top and Bottom {top_n_features} Indicative Features for {class_name}')
    plt.tight_layout()  # Fit the plot neatly
    plt.show()

# Display the feature importances for each class
for class_index in [0, 1]:  # Assuming binary classification for Safe Email (0) and Phishing Email (1)
    plot_feature_importance(class_index, feature_log_prob, feature_names, top_n_features)
# Plot the Precision-Recall (PR) curve
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
# Assuming `model` is your trained model and X_test is your test dataset
# Predict probabilities for the positive class
y_scores = model.predict_proba(X_test)[:, 1]
# Calculate precision and recall for all thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the Precision-Recall curve
display = PrecisionRecallDisplay(precision=precision, recall=recall)
display.plot()

plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Plot the Cumulative Gain Curve
def plot_cumulative_gain(y_true, y_scores):
    # Sort scores and corresponding truth values
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Calculate the cumulative sum of the true positive instances
    cum_true_positives = np.cumsum(y_true_sorted)
    # Normalize by the total number of positives to get the cumulative gains
    cum_gains = cum_true_positives / cum_true_positives[-1]

    # Calculate the baseline (random model's performance)
    baseline = np.linspace(0, 1, len(cum_gains))

    # Plotting the Cumulative Gains
    plt.figure(figsize=(10, 6))
    plt.plot(cum_gains, label='Model')
    plt.plot(baseline, label='Baseline', linestyle='--')
    plt.title('Cumulative Gains Curve')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Cumulative gain')
    plt.legend()
    plt.show()

plot_cumulative_gain(y_test, y_scores)


# _________________________Saving model starts here_________________________
# https://www.kaggle.com/datasets/phangud/spamcsv


# Save the model to a file
dump(model, 'MNB_trained_model.joblib')

# Load the model from the file
model = load('MNB_trained_model.joblib')
vectorizer = load('MNB_vectorizer.joblib')

# Now I can use `model` to make predictions on new data
# https://www.kaggle.com/datasets/phangud/spamcsv

# Loading the phishing email dataset
print("Loading dataset 2...")
# Reading the dataset from a CSV file
df_new = pd.read_csv("Phishing_SMS.csv", delimiter=';')
# Displaying the first few entries of the dataset to verify it's loaded correctly
print("Dataset successfully loaded. Displaying the first few entries:")
print(df_new.head(), '\n')

# Drop rows with any NaN values
df_new = df_new.dropna()

# Preprocess the new dataset
# Assuming that the new dataset requires the same preprocessing and has a column named 'Email Text'
df_new['Email Text'] = df_new['Email Text'].apply(preprocess_text)  # Use the same preprocessing function
df_new['URLs Count'] = df_new['Email Text'].apply(count_urls)
df_new['Contains HTML'] = df_new['Email Text'].apply(contains_html)
df_new['Length'] = df_new['Email Text'].apply(len)

# Vectorize the new data
X_new = vectorizer.transform(df_new['Email Text']).toarray()

df_new['Email Type'] = df_new['Email Type'].replace(['Safe Email', 'Phishing Email'], [0, 1])
y_new = df_new['Email Type'].values

y_new = y_new.astype(int)
print("Unique values in y_new:", np.unique(y_new))
from imblearn.over_sampling import SMOTE

# Applying SMOTE to the vectorized new data
smote = SMOTE(random_state=42)
X_new_resampled, y_new_resampled = smote.fit_resample(X_new, y_new)

print("After SMOTE:")
print("Unique values in y_new_resampled:", np.unique(y_new_resampled))

# Predict using the loaded model on the SMOTE-resampled new data
y_pred_new_resampled = model.predict(X_new_resampled)

# Evaluation on the SMOTE-resampled new data
print(classification_report(y_new_resampled, y_pred_new_resampled))
print("Confusion matrix output:")
print(confusion_matrix(y_new_resampled, y_pred_new_resampled))