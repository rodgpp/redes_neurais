from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

stop = set(stopwords.words('english')) 


#data1 = pd.read_csv('~/dados/dados2/texto_History_Civics.csv')
#data = data1[data1.approved == 1]

#data = pd.read_csv('~/dados/dados2/texto_Warmth_Care_Hunger.csv')

data = pd.read_csv('~/dados/consolidated_text.txt',delimiter=';',names=['id','approved','group','stem'])



y1 = data.group
X1 = data.drop('group', axis=1)

train, test, y_train, y_test = train_test_split(X1, y1,test_size=0.2)


cols22=["id","approved"]
submission=train[cols22]
submission['approved']=submission['approved'].astype(np.float32)

train['len_stem'] = train['stem'].apply(lambda x: len(x))
train['words_stem'] = train['stem'].apply(lambda x: len(x.split()))

test['len_stem'] = test['stem'].apply(lambda x: len(x))
test['words_stem'] = test['stem'].apply(lambda x: len(x.split()))

vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 3), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['stem'])
train_project_stem = vectorizer.transform(train['stem'])
test_project_stem = vectorizer.transform(test['stem'])



cols_to_normalize = ['len_stem', 'words_stem']

scaler = StandardScaler()
for col in cols_to_normalize:
    #print(col)
    scaler.fit(train[col].values.reshape(-1, 1))
    train[col] = scaler.transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))

X = train.drop(['id', 'approved','stem'], axis=1)
#X = train.drop(['id', 'approved','stem','Unnamed: 0'], axis=1)
y = train['approved']
#X_test = test.drop(['id', 'approved','stem','Unnamed: 0'], axis=1)
X_test = test.drop(['id', 'approved','stem'], axis=1)



X_full = csr_matrix(hstack([X.values, train_project_stem]))
X_test_full = csr_matrix(hstack([X_test.values, test_project_stem]))

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, random_state=42)

#-------------------------- XGBoots
params = {'eta': 0.05, 'max_depth': 15, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42, 'silent': True, 'colsample':0.9}
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1000,  watchlist, verbose_eval=10, early_stopping_rounds=20)

modelos = model.predict(xgb.DMatrix(X_test_full), ntree_limit=model.best_ntree_limit)


#------------------------LightGBM
params = {'learning_rate': 0.05, 'max_depth': 14, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'is_training_metric': True, 'seed': 42}
model2 = lgb.train(params, lgb.Dataset(X_train, label=y_train), 1000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=10, early_stopping_rounds=20)






