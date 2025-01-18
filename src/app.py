import os
import pickle
import numpy as  np 
import pandas as pd
from flask import Flask, render_template, request
import warnings

model_dir = os.path.join(os.pardir,'disease_pred','src','model')

app = Flask(__name__)

# Define your list of symptoms
with open(os.path.join(model_dir,'features.pkl'), 'rb') as features:
    symthoms_ = pickle.load(features).columns.values
    symthoms_ = sorted(symthoms_)
    symthoms_ = [i.replace('_',' ').capitalize() for i in symthoms_]
    
def load_encoder_and_models():
    # Load models from files
    with open(os.path.join(model_dir,'rf_model.pkl'), 'rb') as rf_file:
        loaded_rf_model = pickle.load(rf_file)
        
    with open(os.path.join(model_dir,'svm_model.pkl'), 'rb') as svm_file:
        loaded_svm_model = pickle.load(svm_file)

    with open(os.path.join(model_dir,'nb_model.pkl'), 'rb') as nb_file:
        loaded_nb_model = pickle.load(nb_file)
        
    with open(os.path.join(model_dir,'encoder.pkl'), 'rb') as encoder:
        loaded_encoder = pickle.load(encoder)
        
    with open(os.path.join(model_dir,'features.pkl'), 'rb') as features:
        loaded_features = pickle.load(features)

    return loaded_rf_model, loaded_svm_model, loaded_nb_model,loaded_encoder, loaded_features


# Define the predictDisease function using the loaded encoder and models
def predictDisease(symptoms):
    loaded_rf_model, loaded_svm_model, loaded_nb_model, loaded_encoder, loaded_features =  load_encoder_and_models()
    
# Define the symptom index dictionary and predictions classes
    symptoms_list = loaded_features.columns.values

    symptom_index = {symptom.capitalize(): index for index, symptom in enumerate(symptoms_list)}
    predictions_classes = loaded_encoder.classes_

    data_dict = {
        "symptom_index": symptom_index,
        "predictions_classes": predictions_classes
                }
        
    symptoms = symptoms.split(",")
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
   
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.capitalize(), -1)
        if index != -1:
            input_data[index] = 1
  
    # Reshape the input data and convert it into a suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)
    
    column_names = loaded_features.columns

    # Convert NumPy array to Pandas DataFrame with specified column names
    input_data = pd.DataFrame(input_data, columns=column_names)
 
    # Generate individual outputs
    rf_prediction = data_dict["predictions_classes"][loaded_rf_model.predict(input_data)]
    svm_prediction = data_dict["predictions_classes"][loaded_svm_model.predict(input_data)]
    nb_prediction = data_dict["predictions_classes"][loaded_nb_model.predict(input_data)]
    
    all_predictions = [rf_prediction, svm_prediction, nb_prediction]
    unique_predictions, counts = np.unique(all_predictions, return_counts=True)
    final_prediction = unique_predictions[np.argmax(counts)]
    
    return final_prediction

# Route to render the HTML template with dynamic symptoms and handle form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None  
    full_name = None 
    selected_symptoms = None  
    
    if request.method == 'POST':
        # Extract form data
        full_name = request.form['fullName'].capitalize()
        age = request.form['age']
        selected_symptoms = request.form.getlist('symptoms')
        selected_symptoms = ','.join(selected_symptoms)
        model_form_sym = selected_symptoms.replace(' ','_').lower()
        
        # Call the function to process the data and get the outcome
        result = predictDisease(model_form_sym)

    # Render the HTML template with dynamic symptoms for initial GET request
    return render_template('index.html', symptoms=symthoms_, result=result, name =full_name, selected_symptoms=selected_symptoms)

if __name__ == '__main__':
    app.run(debug=True)

