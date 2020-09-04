import pandas as pd
from sklearn import model_selection

class CrossValidation():
    def __init__(
        self,
        df,
        target_cols,
        num_folds=5,
        problem_type = 'binary_classification',
        shuffle=True,
        random_state=42
        ):
        self.dataFrame = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.num_folds = num_folds
        self.problem_type = problem_type
        self.shuffle=shuffle
        self.random_state = random_state
    
    def split(self):

        if self.problem_type in ['binary_classification','multiclass_classification']:
            target = self.target_cols[0]
            unique_vals = self.dataFrame[target].nunique()
            if unique_vals ==1:
                raise Exception("Only one unique value found !")
            elif unique_vals >1:
                if self.shuffle:
                    self.dataFrame = self.dataFrame.sample(frac=1).reset_index(drop=True)
                
                self.dataFrame["kfold"] = -1
                skf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                      shuffle=False)

                for FOLD,(train_idx,val_idx) in enumerate(skf.split(X=self.dataFrame,y=self.dataFrame[target].values)):
                    self.dataFrame.loc[val_idx,'kfold']=FOLD

        return self.dataFrame

if __name__ == "__main__":
    df = pd.read_csv('../inputs/train.csv')
    cv = CrossValidation(df,target_cols=["target"])
    df_split=cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())



