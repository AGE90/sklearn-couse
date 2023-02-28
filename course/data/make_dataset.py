import pandas as pd

class MakeDataset:
    
    def read_from_csv(self, path: str) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        path : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        
        return pd.read_csv(path)
    
    