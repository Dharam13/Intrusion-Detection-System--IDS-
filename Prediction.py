import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import joblib
import pickle
import os


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, hid_feats)
        self.lin = torch.nn.Linear(hid_feats, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.lin(x), dim=1)


import os
import pickle
import joblib
import torch

def load_model_and_preprocessors():
    """Load the trained model and all preprocessors"""
    print("Loading model and preprocessors...")

    # Define base path
    base_path = "Saved_Model"

    # Check if all required files exist
    required_files = [
        'gcn_model_complete.pth',
        'scaler.pkl',
        'label_encoder.pkl',
        'model_metadata.pkl'
    ]

    for file in required_files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file '{file}' not found in {base_path}. Please run train_model.py first.")

    # Load model metadata
    with open(os.path.join(base_path, 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # Load preprocessors
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))

    # Load the complete model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join(base_path, "gcn_model_complete.pth"), map_location=device, weights_only=False)
    model.eval()

    print(f"Model loaded successfully on device: {device}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class names: {metadata['class_names']}")

    return model, scaler, label_encoder, metadata, device


def preprocess_data(df, scaler, feature_columns):
    """Preprocess the input data"""
    print("Preprocessing data...")

    # Clean data
    df.columns = df.columns.str.strip()
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Extract features
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Create missing features with zeros
        for feature in missing_features:
            df[feature] = 0

    # Select and order features correctly
    X = df[feature_columns].values

    # Scale features
    X_scaled = scaler.transform(X)

    return X_scaled


def create_graph(X_scaled, k=10):
    """Create graph structure from features"""
    print(f"Creating graph with k={k} neighbors...")

    # Create k-NN graph
    A = kneighbors_graph(X_scaled, n_neighbors=min(k, len(X_scaled) - 1),
                         mode='connectivity', include_self=False).tocoo()
    edge_index = torch.tensor([A.row, A.col], dtype=torch.long)

    # Create PyTorch Geometric data object
    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float),
        edge_index=edge_index
    )

    print(f"Graph created: {data.num_nodes} nodes, {data.num_edges // 2} undirected edges")
    return data


def predict_attacks(csv_file_path, output_file=None):
    """
    Predict attack types from a CSV file

    Args:
        csv_file_path (str): Path to the CSV file containing network traffic data
        output_file (str, optional): Path to save predictions. If None, returns predictions

    Returns:
        DataFrame with original data and predictions
    """

    # Load model and preprocessors
    model, scaler, label_encoder, metadata, device = load_model_and_preprocessors()

    # Load data to predict
    print(f"Loading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"Data shape: {df.shape}")

    # Keep original data for output
    df_original = df.copy()

    # Preprocess data
    X_scaled = preprocess_data(df, scaler, metadata['feature_columns'])

    # Create graph
    data = create_graph(X_scaled, k=metadata['k_neighbors'])
    data = data.to(device)

    # Make predictions
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1).cpu().numpy()
        probabilities = torch.softmax(out, dim=1).cpu().numpy()

    # Convert predictions back to labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Get confidence scores (max probability for each prediction)
    confidence_scores = np.max(probabilities, axis=1)

    # Create results DataFrame
    results_df = df_original.copy()
    results_df['Predicted_Label'] = predicted_labels
    results_df['Confidence'] = confidence_scores

    # Add individual class probabilities
    for i, class_name in enumerate(metadata['class_names']):
        results_df[f'Prob_{class_name}'] = probabilities[:, i]

    # Print summary
    print("\nPrediction Summary:")
    print(f"Total samples: {len(predictions)}")
    print("\nPredicted label distribution:")
    print(pd.Series(predicted_labels).value_counts())

    print(f"\nAverage confidence: {np.mean(confidence_scores):.4f}")
    print(f"Min confidence: {np.min(confidence_scores):.4f}")
    print(f"Max confidence: {np.max(confidence_scores):.4f}")

    # Save results if output file is specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return results_df


def predict_single_sample(sample_data):
    """
    Predict attack type for a single sample

    Args:
        sample_data (dict or pandas.Series): Single sample data

    Returns:
        dict with prediction results
    """

    # Convert to DataFrame if needed
    if isinstance(sample_data, dict):
        df = pd.DataFrame([sample_data])
    elif isinstance(sample_data, pd.Series):
        df = pd.DataFrame([sample_data])
    else:
        raise ValueError("sample_data must be a dict or pandas Series")

    # Use the main prediction function
    results = predict_attacks("temp_single_sample.csv")

    # Return results for single sample
    result = {
        'predicted_label': results.iloc[0]['Predicted_Label'],
        'confidence': results.iloc[0]['Confidence']
    }

    # Add class probabilities
    for col in results.columns:
        if col.startswith('Prob_'):
            result[col.lower()] = results.iloc[0][col]

    return result


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python predict_model.py <csv_file_path>")
        print("Example: python predict_model.py dos.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        sys.exit(1)

    # Make predictions
    try:
        output_file = f"predictions_{os.path.basename(csv_file)}"
        results = predict_attacks(csv_file, output_file)

        print(f"\nFirst 5 predictions:")
        print(results[['Predicted_Label', 'Confidence']].head())

        # Show some statistics
        print(f"\nDetailed prediction statistics:")
        for label in results['Predicted_Label'].unique():
            count = sum(results['Predicted_Label'] == label)
            avg_conf = results[results['Predicted_Label'] == label]['Confidence'].mean()
            print(f"{label}: {count} samples, avg confidence: {avg_conf:.4f}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback

        traceback.print_exc()