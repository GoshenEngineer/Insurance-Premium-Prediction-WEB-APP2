import pickle
import pandas as pd


#importing the machineLearning model

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

#ML_FLOW
MODEL_VERSION = '1.0.0'

#Get class labels from model (important for matching probabilities to class namaes)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    #predict the class
    predicted_class = model.predict(df)[0]

    #Get Probalities for al classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    #create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))


    return {
        'predicted_category': predicted_class,
        'confidence': round(confidence, 4),
        'class_probabilities':class_probs
    }
    