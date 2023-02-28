import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from course.models.export_model import ExportModel

class TrainModel:
    
    def __init__(self) -> None:
        self.estimators = {
            'SVR' : SVR(),
            'GRAD_BOOSTING' : GradientBoostingRegressor()
        }
        
        self.param_grids = {
            'SVR' : {
                'kernel' : ['linear', 'poly', 'rbf'],
                'gamma' : ['auto', 'scale'],
                'C' : [1, 5, 10]
            },
            'GRAD_BOOSTING' : {
                'loss' : ['squared_error', 'absolute_error'],
                'learning_rate' : [0.01, 0.05, 0.1]
            }
        }
        
    def grid_training(self, X: pd.DataFrame, y: pd.Series) -> None:
        
              
        best_score = 999
        best_model = None
        
        for name, estimator in self.estimators.items():
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=self.param_grids[name],
                cv=3
            ).fit(X, y)
            
            score = np.abs(grid.best_score_)
            
            if score < best_score:
                best_score = score
                best_model = grid.best_estimator_
        
        export = ExportModel()
        export.model_export(best_model, best_score)