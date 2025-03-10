import os
import shutil
import random
from collections import defaultdict


def sample_dataset(dataset_path, sample_size, labels, output_path, copy=False):
    os.makedirs(output_path, exist_ok=True)
    sample = defaultdict(list)
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        if not os.path.exists(label_path):
            continue
        label_output_path = os.path.join(output_path, label)
        os.makedirs(label_output_path, exist_ok=True)
        # Get all files in the label path and shuffle them
        files = os.listdir(label_path)
        random.shuffle(files)
        default_sample_size = sample_size if len(files) > sample_size else len(files)

        sample[label] = [os.path.join(label_path, file) for file in files[:default_sample_size]]

        if copy: 
            for i in range(default_sample_size):
                file = files[i]
                shutil.copy(os.path.join(label_path, file), os.path.join(label_output_path, file))
    
    for key, value in sample.items():
        print(key, len(value))
        

    return sample

def merge_defaultdicts(d1, d2):
    merged = defaultdict(list)

    for key, value in d1.items():
        merged[key].extend(value)  # Add values from first dict

    for key, value in d2.items():
        merged[key].extend(value)  # Add values from second dict

    return merged

def save_sample(sample, output_path):
    os.makedirs(output_path, exist_ok=True)
    for label, files in sample.items():
        label_output_path = os.path.join(output_path, label)
        os.makedirs(label_output_path, exist_ok=True)
        for file in files:
            shutil.copy(file, label_output_path)

if __name__ == "__main__":
    # dataset_path = "nuscenes_dataset"
    # sample_size = 500
    # labels = ["Bicycle", "Car", "Motorcycle", "Pedestrian", "Truck", "ScooterRider", "Wheelchair", "Bus"]
    output_path = "sampled_objects2"
    sample_size = 1500
    # sample1 = sample_dataset(dataset_path, sample_size, labels, output_path, copy=False)
    dataset_path = "cropped_objects"

    labels = [label for label in sorted(os.listdir("cropped_objects")) if os.path.isdir(os.path.join("cropped_objects", label))]

    sample2 = sample_dataset(dataset_path, sample_size, labels, output_path, copy=True)
    # merged_sample = merge_defaultdicts(sample1, sample2)
    # print(merged_sample)

