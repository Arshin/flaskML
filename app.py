from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
cols = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [np.float(x) for x in request.form.values()]
    print('int_features are ',int_features)
    final = np.array(int_features)
    # final_df = pd.DataFrame([final], columns=cols)
    prediction = model.predict([final])
    return render_template('home.html', pred='Expected prediction will be {}'.format(prediction))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)