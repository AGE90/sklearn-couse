import joblib
import numpy as np
import course.utils.paths as path

from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array(
        [1.616463184,
         1.53352356,
         0.796666503,
         0.635422587,
         0.362012237,
         0.315963835,
         2.277026653]
    )
    
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediction' : list(prediction)})

if __name__=="__main__":
    
    models_DIR = path.models_dir('best_model.pkl')
    model = joblib.load(models_DIR)
    
    app.run(port=8080)