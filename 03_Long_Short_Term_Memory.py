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
print(tf.__version__)

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
vectorization_method = 'tfidf'

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

# Here I save the fitted vectorizer to disk
dump(current_vectorizer, 'LSTM_vectorizer.joblib')

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


# The decision to exclude certain features is based on preliminary model performance evaluations.

# Tokenize text
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['Email Text'])
sequences = tokenizer.texts_to_sequences(df['Email Text'])
X = pad_sequences(sequences, maxlen=250)

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['Email Type'])
y = to_categorical(integer_encoded)

# Splitting the Dataset into Training and Testing Subsets
# Ensure to split only once, right here, using your tokenized and padded data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5001, output_dim=128, input_length=X.shape[1]))  # Adjust output_dim
model.add(SpatialDropout1D(0.3))  # Increased dropout rate
model.add(LSTM(150, dropout=0.25, recurrent_dropout=0.25))  # Adjusted number of units and dropout
model.add(Dense(y.shape[1], activation='softmax'))  # Adjust the number of neurons to match the one-hot encoded label size

optimizer = Adam(learning_rate=0.001)  # Adjust learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit the model
epochs = 2
batch_size = 128

training_start_time = time.time()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Training duration
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f"Training completed in {training_time:.3f} seconds")

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test accuracy: {accuracy*100}")

# Predicting on the test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report and confusion matrix
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)

# Confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix for Model: LSTM with ({vectorization_method.upper()})')
plt.show()

# Learning curve visualization
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title(f'Model accuracy for Model: LSTM with ({vectorization_method.upper()})')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Loss curve visualization
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title(f'Model loss for Model: LSTM with ({vectorization_method.upper()})')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot the Precision-Recall (PR) curve
y_scores = model.predict(X_test)[:, 1]
y_true_binary = np.argmax(y_test, axis=1)  # Convert from one-hot to binary if necessary

# Now calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)

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
dump(model, 'LSTM_trained_model.joblib')

# Load the model from the file
model = load('LSTM_trained_model.joblib')
vectorizer = load('LSTM_vectorizer.joblib')

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
df_new['Email Text'] = df_new['Email Text'].apply(preprocess_text)
df_new['URLs Count'] = df_new['Email Text'].apply(count_urls)
df_new['Contains HTML'] = df_new['Email Text'].apply(contains_html)
df_new['Length'] = df_new['Email Text'].apply(len)

# Tokenize and pad the text data
sequences_new = tokenizer.texts_to_sequences(df_new['Email Text'])
X_new_padded = pad_sequences(sequences_new, maxlen=250)

# Convert labels to one-hot encoding
df_new['Email Type'] = df_new['Email Type'].replace(['Safe Email', 'Phishing Email'], [0, 1])
integer_encoded_new = label_encoder.transform(df_new['Email Type'])
y_new_categorical = to_categorical(integer_encoded_new)

# Predicting and evaluating the model on the new dataset
y_pred_new = model.predict(X_new_padded)

# Evaluate the model's performance on the new dataset
loss, accuracy = model.evaluate(X_new_padded, y_new_categorical)
print(f'Accuracy on the new dataset: {accuracy*100:.2f}%')

y_pred_labels = np.argmax(y_pred_new, axis=1)
# Convert one-hot encoded true labels back to a single label
y_true_labels = np.argmax(y_new_categorical, axis=1)
# Calculate the confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Optionally, display the confusion matrix with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Safe Email', 'Phishing Email']))
disp.plot(cmap=plt.cm.Blues)
plt.show()
