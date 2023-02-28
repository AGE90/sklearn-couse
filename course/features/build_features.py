from typing import Tuple
import pandas as pd

class BuildFeatures:
    
    def features_target_split(
        self, dataset: pd.DataFrame, 
        drop_cols: list, 
        target: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """_summary_

        Parameters
        ----------
        dataset : pd.DataFrame
            _description_
        drop_cols : list
            _description_
        target : str
            _description_

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            _description_
        """
        
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[target]
        return X, y