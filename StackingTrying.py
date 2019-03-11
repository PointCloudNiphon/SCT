# all packages
# coding: utf-8

from sklearn.metrics import roc_auc_score, roc_curve,mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import warnings
import logging
import pickle
import os
import gc
gc.enable()
# stacking fun
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
	"""
	这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
	x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
	如果输入为pandas的DataFrame类型则会把报错
	"""
	train_num, test_num = x_train.shape[0], x_test.shape[0]
	second_level_train_set = np.zeros((train_num,))
	second_level_test_set = np.zeros((test_num,))
	test_nfolds_sets = np.zeros((test_num, n_folds))
	kf = KFold(n_splits=n_folds)

	for i,(train_index, test_index) in enumerate(kf.split(x_train)):
		print("Fold {}".format(i))
		x_tra, y_tra = x_train[train_index], y_train[train_index]
		x_tst, y_tst =  x_train[test_index], y_train[test_index]
		clf.fit(x_tra, y_tra,eval_set=[(x_tst, y_tst)],verbose=1000, early_stopping_rounds=1000)
		tpred = clf.predict(x_tst)

		# print("CV score: {:<8.5f}".format(roc_auc_score(y_tst, tpred)))
		second_level_train_set[test_index] = tpred
		test_nfolds_sets[:,i] = clf.predict(x_test)

	second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
	return second_level_train_set, second_level_test_set


# model parameters

# load data and transfer 
if __name__ == '__main__':

	modellgb = lgb.LGBMClassifier(max_depth=-1,n_estimators=999999,learning_rate=0.02,colsample_bytree=0.3,num_leaves=2,metric='auc',objective='binary',n_jobs=-1)
	modelxgb = xgb.XGBClassifier(max_depth=2,n_estimators=999999,colsample_bytree=0.3,learning_rate=0.02,objective='binary:logistic',n_jobs=-1)
	modelcb = cb.CatBoostClassifier(iterations=999999,max_depth=2,learning_rate=0.02,colsample_bylevel=0.03,objective="Logloss")
	train_path = '../input/train.csv'
	test_path  = '../input/test.csv'
    
	lgb_path = './lgb_models_stack/'
	xgb_path = './xgb_models_stack/'
	cb_path  = './cb_models_stack/'

#Create dir for models
	for filename in [lgb_path,xgb_path,cb_path]:
		if os.path.exists(filename)==False:
			os.mkdir(filename)
	print('Load Train Data.')
	train_x = pd.read_csv(train_path)
	print('\nShape of Train Data: {}'.format(train_x.shape))
	train_y = np.array(train_x['target'])                                         
	train_x.drop(['ID_code', 'target'], axis=1, inplace=True)
	train_x=np.array(train_x);
	print('Load Test Data.')
	test_x = pd.read_csv(test_path)
	                                     
	test_x.drop(['ID_code'], axis=1, inplace=True)
	print('\nShape of Test Data: {}'.format(test_x.shape))  
	test_x=np.array(test_x)
	train_sets = []
	test_sets = []
	count = 1
	for clf in [modellgb, modelxgb, modelcb]:
		print("\nmodel:{}".format(count))
		count=count+1
		train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
		train_sets.append(train_set)
		test_sets.append(test_set)

	meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
	meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

	#使用决策树作为我们的次级分类器
	print("second_training")
	from sklearn.tree import DecisionTreeClassifier
	dt_model = DecisionTreeClassifier()
	dt_model.fit(meta_train, train_y)
	df_predict = dt_model.predict(meta_test)
	df_predict.to_excel("Stacking.csv")
	# print(df_predict)