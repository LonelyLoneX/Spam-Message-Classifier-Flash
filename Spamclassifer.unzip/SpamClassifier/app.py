
import os
from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define file paths for the model, vectorizer, and dataset
MODEL_PATH = 'spam_classifier_model.pkl'
VECTORIZER_PATH = 'count_vectorizer.pkl'
DATASET_PATH = 'SMSSpamCollection.txt'

def create_fallback_data():
    """Create a larger fallback dataset with realistic spam and ham messages."""
    spam_messages = [
        "WINNER! You have won £1000! Call now to claim your prize!",
        "Click here to claim your FREE gift card worth $500!",
        "URGENT! Your account will be closed. Click link to verify now.",
        "Congratulations! You've been selected for a special offer!",
        "FREE! Reply STOP to unsubscribe from our amazing deals!",
        "WIN BIG! Play our lottery and win cash prizes instantly!",
        "Limited time offer! Get 90% off on luxury watches!",
        "ALERT: Suspicious activity detected. Verify account immediately.",
        "You've won a brand new iPhone! Claim it now before it expires!",
        "Act fast! Special discount expires tonight. Buy now!",
        "Your loan has been approved! Get money in 24 hours!",
        "FINAL NOTICE: Your subscription will expire. Renew now!",
        "Get rich quick! Join our investment program today!",
        "FREE ringtones! Text TONE to 12345 to download now!",
        "BREAKING: You qualify for a $10,000 loan. Apply now!"
    ]
    
    ham_messages = [
        "Hey, how was your day? Hope you're doing well!",
        "Thanks for lunch today. It was great catching up with you.",
        "Can you pick up some milk on your way home?",
        "Meeting is scheduled for 3 PM in the conference room.",
        "Happy birthday! Hope you have a wonderful day!",
        "Don't forget about dinner with mom and dad tomorrow.",
        "The presentation went really well. Thanks for your help!",
        "Are you free this weekend? Want to go see a movie?",
        "Good morning! Have a great day at work today.",
        "Thanks for helping me with the project. You're the best!",
        "See you at the gym later. Don't forget your water bottle.",
        "The weather is beautiful today. Perfect for a walk!",
        "I'll be running a few minutes late for our meeting.",
        "Hope you feel better soon. Take care of yourself!",
        "Congratulations on your promotion! You deserve it!"
    ]
    
    # Create balanced dataset
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    messages = spam_messages + ham_messages
    
    return pd.DataFrame({'label': labels, 'message': messages})

def load_dataset():
    """Load the SMS Spam Collection dataset with multiple fallback strategies."""
    logger.info("Attempting to load SMS Spam Collection dataset...")
    
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset file {DATASET_PATH} not found. Using fallback data.")
        return create_fallback_data()
    
    # Try multiple encoding strategies
    encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            logger.info(f"Trying encoding: {encoding}")
            df = pd.read_csv(
                DATASET_PATH, 
                sep='\t', 
                header=None, 
                names=['label', 'message'],
                encoding=encoding,
                quoting=3,
                on_bad_lines='skip',
                engine='python'  # Use Python engine for better error handling
            )
            
            # Validate the data
            if len(df) < 10:
                logger.warning(f"Dataset too small ({len(df)} rows). Trying next encoding.")
                continue
            
            # Check if we have both spam and ham labels
            unique_labels = df['label'].unique()
            if not ('spam' in unique_labels and 'ham' in unique_labels):
                logger.warning(f"Dataset missing required labels. Found: {unique_labels}")
                continue
            
            logger.info(f"Successfully loaded {len(df)} SMS messages with encoding: {encoding}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed with encoding {encoding}: {str(e)}")
            continue
    
    # If all encodings fail, use fallback data
    logger.warning("All encoding attempts failed. Using fallback data.")
    return create_fallback_data()

def train_and_save_model():
    """Train a Multinomial Naive Bayes model and save it along with the vectorizer."""
    try:
        logger.info("Starting model training...")
        
        # Load the dataset
        df = load_dataset()
        
        # Data preprocessing
        df = df.dropna()  # Remove any NaN values
        df['message'] = df['message'].astype(str)  # Ensure messages are strings
        
        # Convert labels to numerical (0 for ham, 1 for spam)
        df['label'] = df['label'].replace({'ham': 0, 'spam': 1})
        
        # Remove any rows where label mapping failed
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data after preprocessing")
        
        X = df['message']
        y = df['label']
        
        logger.info(f"Training with {len(df)} messages ({sum(y)} spam, {len(y) - sum(y)} ham)")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and fit CountVectorizer
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_test_transformed = vectorizer.transform(X_test)
        
        # Train the Multinomial Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train_transformed, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.3f}")
        
        # Save the trained model and vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        logger.info("Model and vectorizer saved successfully.")
        
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def load_model_and_vectorizer():
    """Load the trained model and vectorizer, train if they don't exist."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Model and vectorizer loaded successfully.")
            return model, vectorizer
        else:
            logger.info("Model files not found. Training new model...")
            return train_and_save_model()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Training new model...")
        return train_and_save_model()

# Initialize model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer()
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    model, vectorizer = None, None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle message classification prediction."""
    try:
        if model is None or vectorizer is None:
            error_msg = "Error: Model not loaded. Please restart the app."
            logger.error(error_msg)
            return render_template('index.html', prediction_text=error_msg, message="")
        
        message = request.form.get('message', '').strip()
        
        if not message:
            return render_template('index.html', 
                                 prediction_text="Please enter a message to classify.", 
                                 message=message)
        
        # Transform the input message using the loaded vectorizer
        message_transformed = vectorizer.transform([message])
        
        # Get prediction and probability
        prediction_label = model.predict(message_transformed)[0]
        prediction_proba = model.predict_proba(message_transformed)[0]
        
        # Map numerical prediction back to text
        prediction_text = "Spam" if prediction_label == 1 else "Not Spam"
        confidence = max(prediction_proba) * 100
        
        result = f"The message is: {prediction_text} (Confidence: {confidence:.1f}%)"
        
        logger.info(f"Prediction: {prediction_text}, Confidence: {confidence:.1f}%")
        
        return render_template('index.html', prediction_text=result, message=message)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        return render_template('index.html', prediction_text=error_msg, message=request.form.get('message', ''))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('index.html', prediction_text="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', prediction_text="Internal server error occurred."), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
