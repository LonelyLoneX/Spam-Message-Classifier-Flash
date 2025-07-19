Spam Message Classifier Web App

This project is a simple web application built with Flask and scikit-learn that classifies SMS messages as either "Spam" or "Not Spam" (Ham). It demonstrates a basic machine learning workflow, from model training to deployment in a web interface.
Table of Contents

    Features

    Technologies Used

    Dataset

    Model Training

    How to Run the Application

    Project Structure

    Screenshots

    Future Improvements

    License

Features

    SMS Classification: Accurately classifies input messages as "Spam" or "Not Spam".

    Web Interface: User-friendly web form for submitting messages.

    Real-time Prediction: Provides instant classification results.

    Persistent Model: The trained machine learning model and vectorizer are saved to disk, so they don't need to be retrained on every application restart.

    Responsive Design: Basic responsive layout using Tailwind CSS for better viewing on various devices.

Technologies Used

    Python: The core programming language.

    Flask: A lightweight web framework for building the web application.

    scikit-learn: A powerful machine learning library for model training (specifically CountVectorizer for text processing and MultinomialNB for classification).

    Pandas: Used for data manipulation and loading the dataset.

    Joblib: For saving and loading the trained machine learning model and vectorizer.

    HTML/CSS (Tailwind CSS): For the web interface and styling.

Dataset

The model is trained on the SMS Spam Collection Dataset. This dataset comprises SMS messages labeled as either "ham" (legitimate) or "spam"). It is a widely used dataset for text classification tasks.

    Source: Typically found on the UCI Machine Learning Repository.

    Format: Tab-separated values (.tsv) with two columns: label and message.

    File Name in Project: SMSSpamCollection.csv (the .csv extension is used for convenience, but the content is tab-separated).

Model Training

The app.py script automatically trains the machine learning model if it hasn't been trained and saved already.

    Data Loading: The SMSSpamCollection.csv file is loaded into a Pandas DataFrame.

    Text Vectorization: sklearn.feature_extraction.text.CountVectorizer is used to convert the text messages into numerical feature vectors (a bag-of-words representation).

    Model Selection: A sklearn.naive_bayes.MultinomialNB classifier is chosen for its effectiveness in text classification tasks.

    Training: The model is trained on the vectorized message data.

    Persistence: The trained MultinomialNB model and the CountVectorizer instance are saved as spam_classifier_model.pkl and count_vectorizer.pkl respectively, using joblib. This allows the application to load the pre-trained model quickly without retraining every time it starts.

How to Run the Application

To run this project, follow these steps:

    Clone the Repository (or Download ZIP):
    If you're using Git:

    git clone https://github.com/YOUR_USERNAME/Spam-Message-Classifier-Flask.git
    cd Spam-Message-Classifier-Flask

    If you downloaded the ZIP from GitHub, unzip it and navigate into the project folder.

    Download the Dataset:

        Go to the UCI Machine Learning Repository.

        Download the SMSSpamCollection file (it's usually a .txt file, but it's tab-separated).

        Rename the downloaded file to SMSSpamCollection.csv and place it in the root directory of your project (the same folder as app.py).

    Install Dependencies:
    Make sure you have Python installed. Then, install the required libraries using pip:

    pip install -r requirements.txt

    The requirements.txt file contains:

    Flask
    scikit-learn
    joblib
    pandas

    Run the Flask Application:
    Navigate to your project's root directory in your terminal or command prompt and run:

    python app.py

    The application will start, and you'll see a message indicating the URL (e.g., http://127.0.0.1:5000/).

    Access the Web App:
    Open your web browser and go to the URL provided in the terminal (e.g., http://127.0.0.1:5000/).

Project Structure

.
├── app.py                  # Main Flask application and ML model logic
├── requirements.txt        # Python dependencies
├── SMSSpamCollection.csv   # The dataset used for training (download separately)
├── spam_classifier_model.pkl # Saved trained ML model (generated after first run)
├── count_vectorizer.pkl    # Saved CountVectorizer (generated after first run)
└── templates/
    └── index.html          # HTML template for the web interface

Future Improvements

    More Advanced Preprocessing: Implement techniques like stemming, lemmatization, and stop-word removal for better text processing.

    Different Models: Experiment with other machine learning models (e.g., SVM, Logistic Regression, deep learning models) for comparison.

    Hyperparameter Tuning: Optimize model performance by tuning hyperparameters using techniques like GridSearchCV.

    User Feedback: Add a feature for users to provide feedback on predictions to further improve the model over time.

    Deployment: Deploy the application to a cloud platform like Heroku, AWS, or Google Cloud for public access.

License

This project is open-source and available under the MIT License.


