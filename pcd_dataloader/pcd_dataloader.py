import os
import pickle
import logging
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PointCloudDataLoader(Dataset):
    def __init__(
        self,
        dataset_path: str,
        txt_dir: str,
        cache_path: str = "pointcloud_cache.pkl",
        use_cache: bool = True,
        transform=None,
        objects_ids_dict: dict = None
    ):
        self.dataset_path = dataset_path
        self.txt_dir = txt_dir
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.transform = transform
        self.objects_ids_dict = objects_ids_dict or {
            'Pedestrian': 0,
            'Car': 1,
            'Bicycle': 2,
            'Truck': 3,
            'Motorcycle': 4,
            'Wheelchair': 5,
            'ScooterRider': 6,
            'Bus': 7
        }
        self.data = []  # List to store [length, width, height, point_cloud, label] 

        if self.use_cache and os.path.exists(self.cache_path):
            self.load_from_cache()
        else:
            self.load_dataset()
            self.save_to_cache()

    def load_dataset(self):
        logging.info("Loading dataset from disk...")
        for label_name, label_id in self.objects_ids_dict.items():
            pcd_label_path = os.path.join(self.dataset_path, label_name)
            if not os.path.isdir(pcd_label_path):
                logging.warning(f"Directory for label '{label_name}' not found at path: {pcd_label_path}. Skipping.")
                continue

            for file in os.listdir(pcd_label_path):
                file_path = os.path.join(pcd_label_path, file)
                txt_path = os.path.join(self.txt_dir, label_name, file.replace('.pcd', '.txt'))
                if not os.path.isfile(file_path) or not os.path.isfile(txt_path):
                    logging.warning(f"File {file_path} or {txt_path} is not a valid file. Skipping.")
                    continue

                try:
                    point_cloud = o3d.io.read_point_cloud(file_path)
                    if not point_cloud.has_points():
                        logging.warning(f"Point cloud at {file_path} contains no points. Skipping.")
                        continue

                    # Convert Open3D PointCloud to NumPy array
                    points = np.asarray(point_cloud.points)
                    with open(txt_path, 'r') as f:
                        length, width, height = map(float, f.readline().split())
                    self.data.append([length,width,height, points, label_id])
                except Exception as e:
                    logging.error(f"Error reading point cloud from {file_path} or txt from{txt_path}: {e}. Skipping.")

        logging.info(f"Loaded {len(self.data)} point clouds from disk.")

    def save_to_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.data, f)
            logging.info(f"Dataset cached successfully at '{self.cache_path}'.")
        except Exception as e:
            logging.error(f"Failed to save cache to '{self.cache_path}': {e}.")

    def load_from_cache(self):
        try:
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)
            logging.info(f"Loaded dataset from cache at '{self.cache_path}'. Total samples: {len(self.data)}.")
        except Exception as e:
            logging.error(f"Failed to load cache from '{self.cache_path}': {e}. Proceeding to load from disk.")
            self.load_dataset()
            self.save_to_cache()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}.")

        length, width, height, point_cloud, label = self.data[idx]

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return length, width, height, point_cloud, label
