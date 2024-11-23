# Install required libraries
!pip install pandas scikit-learn nltk ipywidgets

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import ipywidgets as widgets
from IPython.display import display, clear_output

# Download NLTK stopwords
nltk.download('stopwords')


# Load dataset
file_name =  '/content/Articles.csv'  # Get the uploaded file name
data = pd.read_csv(file_name, encoding='latin1')

# Inspect the dataset
print("Dataset Columns:", data.columns)
print(data.head())

# Check if the dataset has a 'Category' column
if 'Category' not in data.columns:
    print("No 'Category' column found. Adding dummy categories.")
    # Assign dummy categories cyclically to match the dataset length
    categories = ['Sports', 'Politics', 'Technology', 'Finance', 'Environment']
    data['Category'] = categories * (len(data) // len(categories)) + categories[:len(data) % len(categories)]

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing to the 'Article' column
data['Article'] = data['Article'].apply(preprocess_text)

# Features and labels
X = data['Article']  # Text data
y = data['Category']  # Category labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Function to predict category for new article
def predict_category(new_article):
    # Preprocess the new article text
    new_article_processed = preprocess_text(new_article)
    # Convert text to TF-IDF features
    new_article_tfidf = vectorizer.transform([new_article_processed])
    # Predict category
    predicted_category = model.predict(new_article_tfidf)[0]
    return predicted_category

# Textbox widget for user to enter article
article_textbox = widgets.Textarea(
    value='',
    placeholder='Type your news article here...',
    description='New Article:',
    layout=widgets.Layout(width='500px', height='150px')
)

# Output widget to display prediction
output = widgets.Output()

# Function to handle input change and display prediction
def on_text_change(change):
    with output:
        clear_output()
        if change.new:
            # Get the predicted category for the entered article
            predicted_category = predict_category(change.new)
            print(f"Predicted Category: {predicted_category}")
        else:
            print("Enter a news article to get the predicted category.")

# Link the textbox input to the on_change function
article_textbox.observe(on_text_change, names='value')

# Display the widgets
display(article_textbox, output)
