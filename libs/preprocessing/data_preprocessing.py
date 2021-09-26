from global_features import Global_features
from task_preprocessing import Preprocess_datasets
from image_sizes import Calcualte_sizes

data_root = '/media/SSD0/ladaza/Data/nnUNet_raw_data'
out_directory = '/media/SSD0/ladaza/Data/Prueba'
num_workers = 50

# Calculate the statistics of the original dataset
# Global_features(data_root, num_workers)

# # Preprocess the datasets
# Preprocess_datasets(out_directory, data_root, num_workers)

# Calculate the dataset sizes (used to plan the experiments)
out_directory = '/media/SSD0/ladaza/Data/Decathlon'
Calcualte_sizes(out_directory, num_workers)
