#!flask/bin/python
import os
from flask import Flask
from flask import request, jsonify
import pandas as pd
from sklearn import linear_model
import joblib
from flask import abort
import json
import train_model
import shutil
from glob import glob


app = Flask(__name__)


@app.route('/isAlive')
def index():
    return 'true'


def get_expected_features():
    n_features = 5
    return ['feat_' + str(i) for i in range(n_features)]


@app.route('/prediction/api/v1.0/getdata', methods=['POST'])
def get_data():
    from sklearn.datasets import make_gaussian_quantiles
    import pandas as pd
    n_samples = 10000
    n_classes = 2
    n_features = 5
    cov = 3
    out_file = 'data/data_raw.csv'

    # obtem os dados de alguma fonte externa...
    x, y = make_gaussian_quantiles(cov=cov, n_samples=n_samples,
                                    n_features=n_features,
                                    n_classes=n_classes,
                                    random_state=1)

    dataset = pd.DataFrame(x, columns= ['feat_' + str(i) for i in range(n_features)])
    dataset['y'] = y
    dataset.to_csv(out_file, index=False)
    return jsonify({'msg': 'dados salvos em {}'.format(out_file)})


@app.route('/prediction/api/v1.0/procdata', methods=['POST'])
def proc_data():
    in_file = 'data/data_raw.csv'
    out_file = 'data_proc/data_proc.csv'

    dataset = pd.read_csv(in_file)
    # pre-processa dados
    # ...
    dataset.to_csv(out_file, index=False)
    return jsonify({'msg': 'dados pre-processados salvos em {}'.format(out_file)})


@app.route('/prediction/api/v1.0/train', methods=['POST'])
def train():
    fl = train_model.train()
    return jsonify({'model_path': fl})


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
    model = None
    model_path = glob('models/model_api/*.pkl')[0]

    try:        
        model = joblib.load(open(model_path, 'rb'))
    except:
        return jsonify({'msg': 'Modelo nao encontrado'})

    prediction = model.predict_proba(data)
    # prob classe 1
    prediction = prediction[:, 1]
    return jsonify(prediction.tolist())


@app.route('/prediction/api/v1.0/validate', methods=['GET'])
def validate_model():
    print('Validando novo modelo...')
    return jsonify({'status': 'ok'})


@app.route('/prediction/api/v1.0/restore', methods=['GET'])
def restore_model():
    print('Usando modelo anterior...')
    return jsonify({'status': 'ok'})


@app.route('/prediction/api/v1.0/setnewmodel', methods=['POST'])
def set_model():
    if not request.json or 'model_path' not in request.json:
        abort(400)

    model_path = request.json['model_path']

    # remove folder com modelo anterior
    folder = 'models/model_api'
    shutil.rmtree(folder)

    # cria folder com novo modelo
    os.makedirs(folder)

    # copia modelo para novo dir
    shutil.copy(model_path, folder)
    return jsonify({'msg': 'modelo salvo em {}'.format(folder)})


if __name__ == '__main__':
    if os.environ['ENVIRONMENT'] == 'production':
        app.run(port=80,host='0.0.0.0')
    if os.environ['ENVIRONMENT'] == 'local':
        app.run(port=5000,host='0.0.0.0')