#!flask/bin/python
import os
from flask import Flask
from flask import request, jsonify
import pandas as pd
from sklearn import linear_model
import pickle
from flask import abort
import json

# creating and saving some model
reg_model = linear_model.LinearRegression()
reg_model.fit([[1.,1.,5.], [2.,2.,5.], [3.,3.,1.]], [0.,0.,1.])
pickle.dump(reg_model, open('some_model.pkl', 'wb'))


app = Flask(__name__)


@app.route('/isAlive')
def index():
    return "true"


def get_expected_features():
    return ['f1', 'f2', 'f3']


@app.route('/prediction/api/v1.0/predict', methods=['POST'])
def get_prediction():
    if not request.json or 'features' not in request.json:
        abort(400)

    data = []
    try:
        feature_names = get_expected_features()        

        for row in request.json['features']:            
            d = {}            
            for feat in feature_names:                
                d[feat] = float(row[feat])
            data.append(d)        
    except:
        abort(400)
    
    data = pd.DataFrame(data)
    loaded_model = pickle.load(open('some_model.pkl', 'rb'))
    prediction = loaded_model.predict(data)
    return jsonify(prediction.tolist())



# @app.route('/prediction/api/v1.0/some_prediction', methods=['GET'])
# def get_prediction2():
#     feature1 = float(request.args.get('f1'))
#     feature2 = float(request.args.get('f2'))
#     feature3 = float(request.args.get('f3'))
#     loaded_model = pickle.load(open('some_model.pkl', 'rb'))
#     prediction = loaded_model.predict([[feature1, feature2, feature3]])
#     return str(prediction)


if __name__ == '__main__':
    if os.environ['ENVIRONMENT'] == 'production':
        app.run(port=80,host='0.0.0.0')
    if os.environ['ENVIRONMENT'] == 'local':
        app.run(port=5000,host='0.0.0.0')