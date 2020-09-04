# export TRAINING_DATA=inputs/train_folds.csv
# export FOLD=0
# export MODEL=$1
import os
import sys

os.environ['TRAINING_DATA'] = 'inputs/train_folds.csv'
os.environ['TEST_DATA'] = 'inputs/test.csv'
#os.environ['FOLD']='0'
# os.environ['MODEL']=sys.argv[1]

# for FOLD in range(5):
#     os.environ['FOLD']= str(FOLD)
#     os.system("python -m src.train")

os.system("python -m src.predict")