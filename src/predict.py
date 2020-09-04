import os
import pandas as pd 
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from . import dispatcher

TEST_DATA = os.environ.get('TEST_DATA')
MODELS = os.environ.get('MODEL')


def predict():

    df_test = pd.read_csv(TEST_DATA)
    test_idx = df_test['id'].values
    predictions = None
    
    for FOLD in range(5):

        df_test = pd.read_csv(TEST_DATA)
        encoder = joblib.load(os.path.join("models",f"{MODELS}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models",f"{MODELS}_{FOLD}_columns.pkl"))
        for c in cols:
            lbl = encoder[c]
            df_test.loc[:,c] = lbl.transform(df_test[c].values.tolist())
        
        clf = joblib.load(os.path.join("models",f"{MODELS}_{FOLD}.pkl"))
        
        preds = clf.predict_proba(df_test[cols])[:,1]
        if FOLD==0:
            predictions = preds
        else:
            predictions +=preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx,predictions)), columns=['id','target'])

    return sub

if __name__ =="__main__":
    submission = predict()
    submission.to_csv(f'models/{MODELS}.csv',index=False)
