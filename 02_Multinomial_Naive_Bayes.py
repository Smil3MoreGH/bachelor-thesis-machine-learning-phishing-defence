import time  # For timing operations
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from imblearn.over_sampling import SMOTE  # For handling class imbalance by oversampling
from nltk.corpus import stopwords  # For removing stopwords from text
from nltk.stem import WordNetLemmatizer  # For lemmatizing words to their base form
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer  # For converting text to vector form
from sklearn.metrics import auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve  # For model evaluation
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split  # For model selection and evaluation
from wordcloud import WordCloud  # For generating a word cloud

import warnings
warnings.filterwarnings('ignore')

import nltk
# Ensuring the necessary NLTK datasets are downloaded
# This is done to avoid redundant downloads and ensure the required data is available for text processing
nltk_data_needed = ['wordnet', 'stopwords']
for dataset in nltk_data_needed:
    # The quiet=True option suppresses output unless the download fails
    if not nltk.download(dataset, quiet=True):
        nltk.download(dataset)

# Loading the phishing email dataset
print("Loading dataset...")
# Reading the dataset from a CSV file
df = pd.read_csv("Phishing_Email.csv")
# Displaying the first few entries of the dataset to verify it's loaded correctly
print("Dataset successfully loaded. Displaying the first few entries:")
print(df.head(), '\n')

# Function to count URLs in an email's text
def count_urls(text):
    text = str(text)  # Ensuring the text is a string
    # This function counts occurrences of "http" and "https" in the text
    # It's a simple heuristic to find URLs in the text
    urls_http = text.count('http')
    # Since "https" is included in "http", I only need to count "http" for both
    return urls_http

# Function to check for HTML content within the text
import re  # Import the regular expressions library
def contains_html(text):
    text = str(text)  # Ensure text is in string format
    # Regular expression to match common HTML tags
    html_pattern = re.compile('<.*?>')  # Matches any text enclosed in angle brackets, which is typical of HTML tags
    # Search for the pattern in the text
    if re.search(html_pattern, text):
        return 1  # Return 1 (True) if any HTML tags are found
    else:
        return 0  # Return 0 (False) if no HTML tags are found

# Adding two new features to the dataset based on the email content
# URLs Count: The number of URLs present in the email text
df['URLs Count'] = df['Email Text'].apply(count_urls)
# Contains HTML: Indicates whether the email contains HTML content
df['Contains HTML'] = df['Email Text'].apply(contains_html)

# Defining the text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert the text to lowercase to standardize it
    text = text.strip()  # Remove leading and trailing whitespace and newlines
    tokens = text.split()  # Split the text into individual words (tokens)
    # Remove stopwords to reduce dimensionality and improve model focus on relevant words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens to reduce them to their base or dictionary form (lemma)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Re-join tokens into a single string for further processing or vectorization
    return ' '.join(tokens)
    # Note: I considered removal of punctuation and numbers, but model performance was better without

# Preprocessing the dataset to prepare for modeling
print("Preprocessing data...")
# Removing unnecessary columns and handling missing values
# 'Unnamed: 0' is often an artifact from data loading and is not needed for analysis
df = df.drop(['Unnamed: 0'], axis=1).dropna()
# Apply the text preprocessing function to clean and standardize the email text data
df['Email Text'] = df['Email Text'].apply(preprocess_text)
# Adding a new feature 'Length' to capture the length of the email text after preprocessing
df['Length'] = df['Email Text'].apply(len)
# Encoding the 'Email Type' to numeric values for model processing
# Safe Email = 0, Phishing Email = 1
df['Email Type'].replace(['Safe Email', 'Phishing Email'], [0, 1], inplace=True)
# Displaying the first few entries of the preprocessed dataset
print("Data preprocessing finished. Displaying the updated first few entries:")
print(df.head(), '\n')

# Creating a word cloud to visualize the most common words in phishing emails
print("Generating word cloud for email text visualization...")
# Concatenate all email texts into a single string to generate the word cloud
all_mails = " ".join(df['Email Text'])
# Generating a word cloud, excluding common stopwords to highlight relevant words
word_cloud = WordCloud(stopwords=set(stopwords.words('english')), width=900, height=700, mode="RGB").generate(all_mails)
# Displaying the word cloud in a 10x6 figure
plt.figure(figsize=(10, 6))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")  # Hiding the axis to focus on the visual representation of word frequency
plt.show()

# Analyzing the distribution of email types before applying techniques to handle class imbalance
print("Analyzing class distribution prior to SMOTE application...")
# Counting instances of each email type to understand the class distribution
class_counts = df['Email Type'].value_counts()
# Visualizing the class distribution as a bar chart
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['Safe Email', 'Phishing Email'])  # Labeling the categories
plt.ylabel('Count')  # Label for the y-axis
plt.title('Email Type Distribution Before SMOTE')  # Chart title
plt.show()

# _________________________Vectorization method selection here_________________________
# Vectorizing email text for feature extraction
# The method of vectorization can be chosen from 'tfidf', 'count', or 'hashing'
vectorization_method = 'count'

# Preparing the dataset 'df' and the target 'y' for modeling
y = df['Email Type'].values
# Depending on the chosen method, instantiate the corresponding vectorizer with predefined parameters
if vectorization_method == 'tfidf':
    # Using TF-IDF Vectorization
    current_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
elif vectorization_method == 'count':
    # Using Count Vectorization
    current_vectorizer = CountVectorizer(stop_words='english', max_features=20000)
elif vectorization_method == 'hashing':
    # Using Hashing Vectorization, noted for its statelessness
    print("Applying HashingVectorizer on email text...")
    current_vectorizer = HashingVectorizer(stop_words='english', n_features=2**15)

# TF-IDF and Count Vectorization require fitting to the dataset, unlike Hashing Vectorization
if vectorization_method in ['tfidf', 'count']:
    X = current_vectorizer.fit_transform(df['Email Text']).toarray()
else:  # For 'hashing', directly transform the text without fitting
    X = current_vectorizer.transform(df['Email Text']).toarray()

# Displaying the dimensions of the feature matrix 'X' and target vector 'y' post-vectorization
print(f"Text data vectorized using {vectorization_method.upper()}. Observing shapes of X and y:")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape, '\n')

# Employing SMOTE for Class Imbalance (Synthetic Minority Over-sampling Technique)
# To address class imbalance by generating synthetic examples of the minority class
print("Implementing SMOTE for class balance enhancement...")
smote = SMOTE(random_state=42)  # Initialize SMOTE with a fixed random state for reproducibility
X_resampled, y_resampled = smote.fit_resample(X, y)  # Apply SMOTE to resample the dataset
# Display the class distribution after applying SMOTE to highlight the balance achieved
print("Post-SMOTE class distribution:")
resampled_class_counts = pd.Series(y_resampled).value_counts()
print(resampled_class_counts, '\n')

# Visualizing Class Distribution After SMOTE
# This step is crucial for understanding how SMOTE has modified the class distribution to mitigate imbalance
print("Visual representation of class distribution following SMOTE enhancement...")
plt.figure(figsize=(8, 6))
plt.bar(resampled_class_counts.index, resampled_class_counts.values, color=['green', 'red'])
plt.xticks(ticks=[0, 1], labels=['Safe Email', 'Phishing Email'])  # Label x-axis with class names for clarity
plt.ylabel('Count')
plt.title('Post-SMOTE Email Type Distribution')
plt.show()

# Splitting the Dataset into Training and Testing Subsets
# This step divides the dataset to ensure that the model can be trained on one set of data and tested on an unseen set.
# It helps evaluate its generalization ability
print("Dividing dataset into training and testing subsets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# Displaying the shapes of the training and testing sets to confirm the split
print("Dataset division completed. Shapes of training and testing sets displayed:")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape, '\n')

# After preprocessing and removing outliers, it's useful to reassess the data statistics
# to understand the impact of preprocessing on the dataset's characteristics.
# Filter the DataFrame to remove emails with more than 25 URLs
filtered_df = df[df['URLs Count'] <= 25]
print("Descriptive statistics for new features after removing emails with >25 URLs:")
print(filtered_df[['URLs Count', 'Contains HTML']].describe(), '\n')
# The preprocessing phase concludes with the data ready for modeling.

# _________________________Model section starts here_________________________

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Cross-validation is employed to establish a baseline model performance.
# This step is crucial for understanding the model's accuracy across different subsets of the dataset.
print("Performing cross-validation to gauge baseline model performance...")
model_cv = MultinomialNB()
cv_scores = cross_val_score(model_cv, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores obtained:", cv_scores)
print("Computed average accuracy score from cross-validation:", cv_scores.mean(), '\n')

# Timing the model training phase to quantify the computation time.
training_start_time = time.time()

# With Naive Bayes, hyperparameter tuning might focus on the 'alpha' parameter for smoothing.
# Note that often the default value of alpha works well, but you can adjust this based on your needs.
print("Exploring alpha parameter with GridSearchCV to refine model performance...")
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')


# Calculating the training duration helps in assessing the efficiency of the training process.
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f"Training completed in {training_time:.3f} seconds")

# With the best parameters found, the model is now ready for predictions.
# This stage evaluates the model's performance on unseen data, simulating how it would perform in real-world scenarios.
model = grid_search.best_estimator_

# Timing the prediction phase to evaluate efficiency.
prediction_start_time = time.time()

# Predicting on the test set and displaying classification metrics provide insights into the model's ability
# to generalize and its performance across different classes.
print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)
print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

# The prediction duration gives an idea of the model's speed in practical applications.
prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time
print(f"Prediction completed in {prediction_time:.3f} seconds")

print("Optimal parameters identified:", grid_search.best_params_)
print("Best scoring accuracy achieved through cross-validation:", grid_search.best_score_, '\n')

# Applying the best parameters found to the model for subsequent predictions and evaluations
model = grid_search.best_estimator_

# Before making predictions, start timing for the prediction process
prediction_start_time = time.time()

# Proceeding to predict on the test set with the optimized model and evaluating the outcomes
print("Conducting predictions on the test dataset...")
y_pred = model.predict(X_test)

# Displaying the results through classification report and confusion matrix for detailed analysis
print("Presenting classification metrics and confusion matrix for assessment:")
print(classification_report(y_test, y_pred))
print("Confusion matrix output:")
print(confusion_matrix(y_test, y_pred), '\n')

# After predictions are made, stop timing and calculate duration
prediction_end_time = time.time()
prediction_time = prediction_end_time - prediction_start_time
print(f"Prediction completed in {prediction_time:.3f} seconds")

# This section visualizes the confusion matrix, learning curve, ROC curve, and feature importance to evaluate and interpret the model's performance.

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
# This illustrates the model's ability to distinguish between the two classes
print("Calculating ROC curve and AUC to evaluate model's discriminative ability...")
# Get the decision function scores for each prediction
y_score = model.decision_function(X_test)
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

# Display the top features for phishing (class 1) and safe emails (class 0)
for i, class_indices in enumerate(top_features_per_class):
    print(f"\nTop features for class {i} ('Phishing Email' if i == 1 else 'Safe Email'):")
    top_features = feature_names[class_indices]
    # Display the feature names
    for feature in top_features:
        print(f"    {feature}")

# Optionally, visualize the top features for a specific class (e.g., phishing emails)
class_index = 1  # 0 for safe email, 1 for phishing email
plt.figure(figsize=(12, 8))
# Sort the top features by their log probability values
sorted_indices = np.argsort(feature_log_prob[class_index, top_features_per_class[class_index]])
# Create a horizontal bar chart
plt.barh(range(top_n_features), feature_log_prob[class_index, top_features_per_class[class_index]][sorted_indices], align='center', color='dodgerblue')
plt.yticks(range(top_n_features), feature_names[top_features_per_class[class_index]][sorted_indices])
plt.xlabel('Log Probability')
plt.title(f'Top {top_n_features} Influential Features for Model: MNB with ({vectorization_method.upper()})')
plt.gca().invert_yaxis()  # Ensure the top features are at the top of the plot
plt.show()
