import course.utils.paths as path
from course.data.make_dataset import MakeDataset
from course.features.build_features import BuildFeatures
from course.models.train_model import TrainModel

if __name__=="__main__":
        
    make_data = MakeDataset()
    data_DIR = path.data_raw_dir('happiness.csv')
    dataset = make_data.read_from_csv(data_DIR)
    
    build_features = BuildFeatures()
    X, y = build_features.features_target_split(
        dataset=dataset,
        drop_cols=['country', 'rank', 'score', 'high', 'low'],
        target='score'
    )
    
    training = TrainModel()
    training.grid_training(X, y)