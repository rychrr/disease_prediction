import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle 
import logging
from datetime import datetime


def set_details():
    """
    Set the details including date formatting, log file path, dataset file path, and directory path for saving models.

    Returns:
    Tuple[str, str]: Tuple containing the dataset file path and the directory path for saving models.
    """
    try:
        # Set the date formatting
        date_format = '%Y_%m_%d'
        current_date = datetime.now().strftime(date_format)

        # Set the log file path
        log_file_path = f"/Users/mac/Desktop/star platinum/projects/olutomiwa/src/logs/log_{current_date}.log"
        logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='a',
                            format=f'%(asctime)s - %(levelname)s - %(message)s')

        # Set the dataset file path and the directory path for saving models
        file_path = '/Users/mac/Desktop/star platinum/projects/olutomiwa/src/datasets/combined_data.csv'
        dir_path = '/Users/mac/Desktop/star platinum/projects/olutomiwa/src/model'
        
        return file_path, dir_path

    except Exception as e:
        logging.error(f"An error occurred while setting details: {str(e)}")
        return None


def train_and_save_models():
    # Set the file path 
    file_path, dir_path =set_details()

    """
    Load the dataset, preprocess, train RandomForest, SVC, and Naive Bayes models,
    save the trained models and LabelEncoder to files.

    Parameters:
    - file_path (str): The path to the input dataset file.
    - dir_path (str): The directory path to save the trained models.

    Returns:
    None
    """
    try:
        # Load the dataset
        concatenated_data = pd.read_csv(os.path.join(file_path), index_col=False).reset_index(drop=True).dropna(axis=1)
        logging.info("Dataset loaded successfully.")


        # Initialize LabelEncoder
        encoder = LabelEncoder()

        # Apply label encoding to the "prognosis" column in the DataFrame
        concatenated_data["prognosis"] = encoder.fit_transform(concatenated_data["prognosis"])

        # Select features (X) and target variable (y)
        X = concatenated_data.iloc[:, :-1]
        y = concatenated_data.iloc[:, -1]

        # Initialize models
        rf_model = RandomForestClassifier(random_state=18)
        svm_model = SVC(probability=True)
        nb_model = GaussianNB()

        # Train models on data
        rf_model.fit(X, y)
        logging.info("Random Forest Model trained successfully.")
        print("Random Forest Model trained successfully.")
        svm_model.fit(X, y)
        logging.info("SVM Model trained successfully.")
        print("SVM Model trained successfully.")
        nb_model.fit(X, y)
        logging.info("Naives Bayes Model trained successfully.")
        print("Naives Bayes Model trained successfully.")
        
        logging.info("All Models trained successfully.")
        print("All Models trained successfully.")

        # Save models to files
        with open(os.path.join(dir_path, 'rf_model.pkl'), 'wb') as rf_file:
            pickle.dump(rf_model, rf_file)
            logging.info("Random Forest model saved.")
            print("Random Forest model saved.")

        with open(os.path.join(dir_path, 'svm_model.pkl'), 'wb') as svm_file:
            pickle.dump(svm_model, svm_file)
            logging.info("SVM model saved.")
            print("SVM model saved.")

        with open(os.path.join(dir_path, 'nb_model.pkl'), 'wb') as nb_file:
            pickle.dump(nb_model, nb_file)
            logging.info("Naive Bayes model saved.")
            print("Naive Bayes model saved.")

        with open(os.path.join(dir_path, 'encoder.pkl'), 'wb') as encoder_file:
            pickle.dump(encoder, encoder_file)
            logging.info("LabelEncoder saved.")
            print("LabelEncoder saved.")
            
        with open(os.path.join(dir_path, 'features.pkl'), 'wb') as feature_file:
            pickle.dump(X, feature_file)
            logging.info("Features saved.")
            print("Matrix Features  saved.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":   
    train_and_save_models()
