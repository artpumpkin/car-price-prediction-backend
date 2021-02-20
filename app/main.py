from os.path import dirname, realpath, abspath, join
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__) 
CORS(app)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello': 'world'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    row = [data['views'], data['mileage'], data['fuel'], data['fiscalPower'], data['carAge'], *data['marque']]

    return jsonify({'price': get_price(row)})


def get_price(X):
    dir_path = dirname(realpath(__file__))

    with open(abspath(join(dir_path, './scaler_minMax.pickle')), 'rb') as f:
        scaler = pickle.load(f)

    with open(abspath(join(dir_path, './RFR_model.pickle')), 'rb') as f:
        RFR = pickle.load(f)

    X = scaler.transform(np.array([0] + X).reshape(-1, 1).T)[:, 1:]

    result = RFR.predict(X).tolist()

    zeros = [0] * 17
    zeros[0] = result[0]

    price = scaler.inverse_transform(np.array(zeros).reshape(-1, 1).T)[0][0]

    return round(price, 2)