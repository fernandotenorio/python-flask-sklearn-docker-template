import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime
import os


def train():
	df = pd.read_csv('data_proc/data_proc.csv')
	y = df['y']
	x = df.drop('y', 1)
	del df

	clf = LogisticRegression(random_state=0).fit(x, y)

	now = datetime.now()
	timestamp = now.strftime('%Y%m%d%H%M%S')
	year_month = now.strftime('%Y%m')
	model_dir = '{}/{}/'.format('models', year_month)

	if not os.path.exists(model_dir):
		os.makedirs(model_dir)


	fl = '{}{}.pkl'.format(model_dir, timestamp)
	joblib.dump(clf, open(fl, 'wb'))
	return fl

