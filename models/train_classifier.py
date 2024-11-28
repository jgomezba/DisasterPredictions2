import sys
import os
import pandas as pd
import re
import nltk
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV

# nltk.download('punkt')

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
    database_filepath: str. Filepath for the database containing the dataset.

    Returns:
    X: DataFrame. Features (messages).
    Y: DataFrame. Target (categories).
    category_names: list. List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('data', engine)
    df.drop(columns=['Message length','ShortLong message'], inplace=True)
    X = df['message']
    Y = df.iloc[:, 4:]  # Assuming the first 4 columns are ID, message, etc.
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning model pipeline and perform hyperparameter tuning with GridSearchCV.

    Returns:
    GridSearchCV. Grid search object with the best model.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define the parameters to tune
    parameters = {
        'tfidf__max_df': [1.0],
        'tfidf__ngram_range': [(1, 1)],  # Unigrams or unigrams + bigrams
        'clf__estimator__n_estimators': [80, 120],  # Number of trees in the forest
        'clf__estimator__min_samples_split': [2, 5],  # Minimum samples required to split a node
    }
    
    # GridSearchCV to find the best parameters
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=3,  # 3-fold cross-validation
        verbose=3,  # Show progress
        n_jobs= -1  # Use all available CPUs
    )
    
    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model.

    Args:
    model: Pipeline. Trained machine learning model.
    X_test: DataFrame. Test features.
    Y_test: DataFrame. Test target values.
    category_names: list. List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Args:
    model: Pipeline. Trained machine learning model.
    model_filepath: str. Filepath for saving the model.
    """
    joblib.dump(model, model_filepath)

def main():
    """
    Main function to run the script.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        model_full_path = os.path.join(os.path.dirname(__file__), model_filepath)
        if not os.path.exists(model_full_path):
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))
            X, Y, category_names = load_data(database_filepath)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)
            
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_full_path))
            save_model(model, model_full_path)

            print('Trained model saved!')
        else:
            print(f"Model was previously generated in {model_full_path}")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    print()
    main()
