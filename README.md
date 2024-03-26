# Disease Prediction Web Application

This is a Flask-based web application that predicts probable diseases based on user-entered symptoms. The application provides a user interface where users can input their symptoms through a form, and upon submission, the application predicts the most probable disease associated with the symptoms.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone < not yet moved to github>
    ```

2. Navigate to the project directory:

    ```bash
    cd disease_pred
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/` to access the web application.

3. Enter your full name, age, and select the symptoms you are experiencing from the dropdown menu.

4. Click the "Submit" button to submit the form.

5. The application will predict the most probable disease based on the symptoms entered and display the result below the form.

6. To clear the form, click the "Clear" button.

## Directory Structure

|-- README.md # Project README and documentation
|-- requirements.txt # Required Python dependencies
|-- src # Source code directory
  |-- app.py # Main Flask application file
  |-- Datasets
    |-- combined_data.csv
    |-- testing.csv
    |-- training.csv
    |-- docs
        |-- Disease Prediction - Docs.docx
        |-- Disease_Prediction_model.drawio
        |-- disease_predictor_worksheet.pdf
        |-- OLUTOMIWA PROJECT PROPOSAL for Machine learning and disease outbreake prediction.docx
        |-- worlflow_desgins.png
    |-- notebooks
        |-- disease_predictor_worksheet.ipynb
    | |-- model # Directory containing ML models and data
    | | |-- features.pkl # Pickled file containing features (symptoms)
    | | |-- encoder.pkl # Pickled file containing encoder
    | | |-- rf_model.pkl # Pickled file containing Random Forest model
    | | |-- svm_model.pkl # Pickled file containing SVM model
    | | |-- nb_model.pkl # Pickled file containing Naive Bayes model
    |-- static # Static files directory
    | |-- style.css # CSS file for styling the web application
    |-- templates # HTML templates directory
    | |-- index.html # HTML template for the web application interface


## Technologies Used

- Python
- Flask
- HTML
- CSS
- Bootstrap

## Credits

- The ML models and data used in this project are sourced from [Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning).
- This project was developed by Olutomiwa.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.mit.edu/~amini/LICENSE.md) file for details.


