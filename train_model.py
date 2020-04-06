import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


def train():
	df = pd.read_csv('data_proc/data_proc.csv')
	y = df['y']
	x = df.drop('y', 1)
	del df

	clf = LogisticRegression(random_state=0).fit(x, y)
	fl = 'models/model.pkl'
	pickle.dump(clf, open(fl, 'wb'))
	return fl

