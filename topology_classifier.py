import numpy as np
import gudhi
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from pcd_dataloader import PointCloudDataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class PointCloudProcessor:
    def __init__(self, normalize=True, num_points=None):
        self.normalize = normalize
        self.num_points = num_points
        self.scaler = StandardScaler()
    
    def process(self, point_cloud):
        """Process a single point cloud."""
        # Convert to numpy if needed
        # if not isinstance(point_cloud, np.ndarray):
        #     points = np.asarray(point_cloud, dtype=np.float32)
        # else:
        #     points = point_cloud
        points = np.asarray(point_cloud, dtype=np.float32)
        
        
        # Normalize if requested
        if self.normalize:
            # Center and scale
            centroid = np.mean(points, axis=0)
            points = points - centroid
            scale = np.max(np.abs(points))
            if scale > 0:
                points = points / scale
        
        # Downsample if needed
        if self.num_points and len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]

        # Optionally check for NaN or Inf:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print("Warning: Found NaN or Inf in the processed point cloud. Clipping or skipping.")
            # Option 1: clip them
            points = np.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Option 2: skip the cloud entirely, return empty or None
            # return None

        return points

class PersistentHomologyFeatures:
    def __init__(self, max_dimension=2, max_edge_length=1.0):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
    
    def compute_persistence(self, points):
        """Compute persistence diagrams for a point cloud."""
        # Create Vietoris-Rips complex
        rips = gudhi.RipsComplex(points=points, max_edge_length=self.max_edge_length)
        
        # Compute persistence
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        persistence = simplex_tree.persistence()
        
        return persistence
    
    def extract_features(self, persistence):
        """Extract features from persistence diagram."""
        features = []
        
        # Process each dimension
        for dim in range(self.max_dimension + 1):
            # Get persistence pairs for this dimension
            pairs = [(birth, death) for (d, (birth, death)) in persistence if d == dim and death != float('inf')]
                        
            if len(pairs) > 0:
                pairs = np.array(pairs)
                
                # Calculate basic statistics
                lifetimes = pairs[:, 1] - pairs[:, 0]
                features.extend([
                    np.mean(lifetimes),
                    np.std(lifetimes),
                    np.max(lifetimes),
                    len(pairs)  # number of features in this dimension
                ])
            else:
                # Add zeros if no features in this dimension
                features.extend([0, 0, 0, 0])
        
        return np.array(features)

class TopologyClassifier:
    def __init__(self, processor=None, feature_extractor=None, n_jobs=-1):
        self.processor = processor or PointCloudProcessor()
        self.feature_extractor = feature_extractor or PersistentHomologyFeatures()
        self.classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
        self.n_jobs = n_jobs

    def _process_single_point_cloud(self, pc):
        """Helper function to process a single point cloud and extract its features."""
        # Process point cloud
        processed_pc = self.processor.process(pc)

        # Compute persistence
        persistence = self.feature_extractor.compute_persistence(processed_pc)

        # Extract features from persistence
        pc_features = self.feature_extractor.extract_features(persistence)
        return pc_features

    def extract_features(self, point_clouds):
        """Extract features from a list of point clouds using parallel processing."""
        # Parallelize over the list of point clouds
        with Parallel(n_jobs=self.n_jobs) as parallel:
        # Perform your parallel calls
            features = parallel(
                delayed(self._process_single_point_cloud)(pc)
                for pc in point_clouds
            )
            parallel._terminate_and_reset()
        
        print("Successfully extracted features")
        return np.array(features)
    
    def fit(self, data, labels):
        """Train the classifier."""
        # Extract features
        point_clouds = data[:, -1]
        X = self.extract_features(point_clouds)

        X = np.hstack([data[:, :-1], X])
        
        # Train classifier
        self.classifier.fit(X, labels)
    
    def predict(self, point_clouds):
        """Predict labels for point clouds."""
        # Extract features
        X = self.extract_features(point_clouds)
        
        # Predict
        return self.classifier.predict(X)

# Deep learning extension
class TopologyNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TopologyNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels, processor=None, feature_extractor=None, n_jobs=-1):
        self.processor = processor or PointCloudProcessor()
        self.feature_extractor = feature_extractor or PersistentHomologyFeatures()
        self.n_jobs = n_jobs
        
        # Pre-compute features
        self.features = self.extract_features(point_clouds)
            
        self.features = torch.FloatTensor(self.features).to("cuda:0")
        self.labels = torch.LongTensor(labels).to("cuda:0")
    
    def _process_single_point_cloud(self, pc):
        """Helper function to process a single point cloud and extract its features."""
        # Process point cloud
        processed_pc = self.processor.process(pc)

        # Compute persistence
        persistence = self.feature_extractor.compute_persistence(processed_pc)

        # Extract features from persistence
        pc_features = self.feature_extractor.extract_features(persistence)
        return pc_features

    def extract_features(self, point_clouds):
        """Extract features from a list of point clouds using parallel processing."""
        # Parallelize over the list of point clouds
        with Parallel(n_jobs=self.n_jobs) as parallel:
        # Perform your parallel calls
            features = parallel(
                delayed(self._process_single_point_cloud)(pc)
                for pc in point_clouds
            )
            parallel._terminate_and_reset()
        
        print("Successfully extracted features")
        return np.array(features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]        


# Example usage
def main():
    # dataset_path = 'merged_sampled_objects'
    dataset_path = 'sampled_objects2'
    txt_path = 'sampled_objects2_txt'
    objects_ids_dict = {k:v for v, k in enumerate(sorted(os.listdir(dataset_path)))}
    data = PointCloudDataLoader(dataset_path, txt_path, objects_ids_dict=objects_ids_dict, use_cache=False).data
    data = []
    labels = []

    for obj in data:
        if np.any(np.isnan(point)) or np.any(np.isinf(point)):
            print("Warning: Found NaN or Inf in the processed point cloud. Clipping or skipping.")
            # Option 1: clip them
            point = np.nan_to_num(point, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Option 2: skip the cloud entirely, return empty or None
            # return None
        length, width, height, point, label = obj
        data.append([length, width, height, point])
        labels.append(label)
    
    
    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(labels))
    # Train classical classifier
    classifier = TopologyClassifier()
    classifier = TopologyClassifier(processor=PointCloudProcessor(num_points=100), n_jobs=-1)
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    
    # Predict
    print("Predicting...")
    predictions = classifier.predict(X_test)
    y_hat = np.array(predictions)
    accuracy = accuracy_score(y_test, y_hat)
    cm = confusion_matrix(y_test, y_hat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pedestrian', 'Car', 'Bicycle', 'Truck', 'Motorcycle', 'Wheelchair', 'ScooterRider', 'Bus'])

    disp.plot()  # This plots the confusion matrix

    # Save the figure
    plt.savefig("my_confusion_matrix.png", dpi=600, bbox_inches='tight')

    # Optionally close the figure if you don't want it displayed:
    plt.close()
    print(f"Accuracy: {accuracy}")

    # Train deep learning model
    # # Create datasets
    # train_dataset = PointCloudDataset(X_train, y_train, processor=PointCloudProcessor(num_points=100))
    # test_dataset = PointCloudDataset(X_test, y_test, processor=PointCloudProcessor(num_points=100))
    # print("Successfully created datasets")
    
    # # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32)
    # print("Successfully created data loaders")
    
    # # Initialize model
    # input_dim = train_dataset.features.shape[1]
    # model = TopologyNet(input_dim=input_dim, num_classes=num_classes).to("cuda:0")
    # # model.load_state_dict(torch.load("model2.pth"))
    # print("Successfully initialized model")
    
    # # Training loop (simplified)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # num_epochs = 250
    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0
    #     for batch, (features, labels) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         outputs = model(features)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     print(f"Epoch {epoch+1}, loss: {epoch_loss / len(train_loader)}")

    # # Evaluation
    # model.eval()
    # y_preds = []
    # y_true = []

    # with torch.no_grad():
    #     for features, lbls in test_loader:
    #         features = features.to("cuda:0")
    #         lbls = lbls.to("cuda:0")
    #         outputs = model(features)
    #         _, predicted = torch.max(outputs, 1)
            
    #         y_preds.extend(predicted.cpu().numpy())
    #         y_true.extend(lbls.cpu().numpy())

    # # Calculate accuracy
    # accuracy = accuracy_score(y_true, y_preds)
    # print(f"Accuracy: {accuracy:.4f}")

    # # Confusion matrix
    # cm = confusion_matrix(y_true, y_preds)
    
    
    # class_names = [label for label in sorted(os.listdir("cropped_objects")) if os.path.isdir(os.path.join("cropped_objects", label))]

    # assert len(class_names) == num_classes, "Number of class names must match number of classes"
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # fig, ax = plt.subplots(figsize=(8, 6))

    # disp.plot(ax=ax, cmap=plt.cm.Blues)
    # plt.xticks(rotation=45, ha='right')  # <-- This ensures x-axis labels don’t overlap
    # plt.tight_layout()                   # <-- Adjusts plot to fit labels
    # plt.title('Confusion Matrix - Deep Learning Model UCLA Dataset')
    # plt.savefig("deep_learning_confusion_matrix_ucla.png", dpi=600, bbox_inches='tight')
    # print("Saved confusion matrix as deep_learning_confusion_matrix_ucla.png")

    # # Save model
    # torch.save(model.state_dict(), "model3.pth")
    # print("Saved model weights to model3.pth")
    
    

    


if __name__ == "__main__":
    main()