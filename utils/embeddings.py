"""
Embedding generation utilities for satellite imagery.

This module provides functions for generating embeddings from satellite imagery.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.ports import get_all_ports

# Default embedding parameters
DEFAULT_EMBEDDING_DIM = 128

def load_image(image_path):
    """
    Load an image from a file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_features_from_image(img_array):
    """
    Extract features from an image.
    
    This function computes:
    1. Basic statistics (mean, std, min, max) for each band
    2. Simple texture features (gradient magnitudes)
    3. Color histograms
    
    Args:
        img_array: Image as numpy array
        
    Returns:
        numpy.ndarray of features
    """
    if img_array is None:
        return None
    
    # For RGB images
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        features = []
        
        # Basic statistics for each band
        for i in range(min(3, img_array.shape[2])):
            band = img_array[:, :, i].astype(float)
            features.extend([
                np.mean(band),
                np.std(band),
                np.min(band),
                np.max(band),
                np.median(band),
                np.percentile(band, 25),
                np.percentile(band, 75)
            ])
        
        # Simple texture features (gradient magnitudes)
        for i in range(min(3, img_array.shape[2])):
            band = img_array[:, :, i].astype(float)
            grad_x = np.gradient(band, axis=0)
            grad_y = np.gradient(band, axis=1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            features.extend([
                np.mean(grad_mag),
                np.std(grad_mag),
                np.max(grad_mag),
                np.median(grad_mag)
            ])
        
        # Color histograms (10 bins per channel)
        for i in range(min(3, img_array.shape[2])):
            band = img_array[:, :, i].astype(float)
            hist, _ = np.histogram(band, bins=10, range=(0, 255), density=True)
            features.extend(hist)
        
        return np.array(features)
    
    # For single band images or other formats
    else:
        # Flatten and normalize
        flat_img = img_array.flatten().astype(float)
        return flat_img / 255.0

def generate_embedding(features, target_dim=DEFAULT_EMBEDDING_DIM):
    """
    Generate a lower-dimensional embedding from image features.
    
    Args:
        features: Image features
        target_dim: Target embedding dimension
        
    Returns:
        numpy.ndarray
    """
    if features is None:
        return None
    
    # If features are already smaller than target_dim, pad with zeros
    if len(features) <= target_dim:
        embedding = np.zeros(target_dim)
        embedding[:len(features)] = features
        return embedding
    
    # Otherwise, use PCA to reduce dimensionality
    pca = PCA(n_components=target_dim)
    embedding = pca.fit_transform(features.reshape(1, -1))[0]
    
    return embedding

def normalize_embedding(embedding):
    """
    Normalize an embedding to unit length.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        numpy.ndarray
    """
    if embedding is None:
        return None
    
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    else:
        return embedding

def generate_embedding_for_image(image_path, target_dim=DEFAULT_EMBEDDING_DIM):
    """
    Generate an embedding for an image.
    
    Args:
        image_path: Path to the image file
        target_dim: Target embedding dimension
        
    Returns:
        numpy.ndarray
    """
    # Load image
    img_array = load_image(image_path)
    if img_array is None:
        return None
    
    # Extract features
    features = extract_features_from_image(img_array)
    if features is None:
        return None
    
    # Generate embedding
    embedding = generate_embedding(features, target_dim)
    if embedding is None:
        return None
    
    # Normalize embedding
    embedding = normalize_embedding(embedding)
    
    return embedding

def generate_embeddings_for_port(port_name, imagery_dir, output_dir, target_dim=DEFAULT_EMBEDDING_DIM):
    """
    Generate embeddings for all images in a port.
    
    Args:
        port_name: Name of the port
        imagery_dir: Directory containing imagery
        output_dir: Directory to save embeddings
        target_dim: Target embedding dimension
        
    Returns:
        Dictionary mapping H3 indices to embeddings
    """
    # Create output directory
    port_imagery_dir = os.path.join(imagery_dir, port_name)
    port_output_dir = os.path.join(output_dir, port_name)
    os.makedirs(port_output_dir, exist_ok=True)
    
    # Check if imagery directory exists
    if not os.path.exists(port_imagery_dir):
        print(f"No imagery directory found for port: {port_name}")
        return {}
    
    # Get all image files
    image_files = [f for f in os.listdir(port_imagery_dir) if f.endswith('.tif')]
    
    if len(image_files) == 0:
        print(f"No imagery found for port: {port_name}")
        return {}
    
    # Generate embeddings
    embeddings = {}
    
    for image_file in tqdm(image_files, desc=f"Generating embeddings for {port_name}"):
        h3_index = os.path.splitext(image_file)[0]
        image_path = os.path.join(port_imagery_dir, image_file)
        
        try:
            embedding = generate_embedding_for_image(image_path, target_dim)
            if embedding is not None:
                embeddings[h3_index] = embedding
                
                # Save embedding
                output_path = os.path.join(port_output_dir, f"{h3_index}.npy")
                np.save(output_path, embedding)
        except Exception as e:
            print(f"Error generating embedding for {h3_index}: {e}")
    
    print(f"Generated {len(embeddings)} embeddings for {port_name}")
    
    # Save metadata
    metadata = {
        'port': port_name,
        'embedding_dim': target_dim,
        'num_embeddings': len(embeddings),
        'h3_indices': list(embeddings.keys())
    }
    
    with open(os.path.join(port_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return embeddings

def generate_embeddings_for_all_ports(imagery_dir, output_dir, target_dim=DEFAULT_EMBEDDING_DIM):
    """
    Generate embeddings for all ports.
    
    Args:
        imagery_dir: Directory containing imagery
        output_dir: Directory to save embeddings
        target_dim: Target embedding dimension
        
    Returns:
        Dictionary mapping port names to embeddings
    """
    all_embeddings = {}
    
    for port_name in get_all_ports():
        port_embeddings = generate_embeddings_for_port(
            port_name, imagery_dir, output_dir, target_dim
        )
        all_embeddings[port_name] = port_embeddings
    
    return all_embeddings

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for satellite imagery")
    parser.add_argument("--port", type=str, help="Port name (tianjin, dalian, shanghai, qingdao, or 'all')")
    parser.add_argument("--imagery_dir", type=str, default="../data/raw", help="Directory containing imagery")
    parser.add_argument("--output_dir", type=str, default="../data/embeddings", help="Output directory for embeddings")
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM, help=f"Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})")
    
    args = parser.parse_args()
    
    if args.port and args.port.lower() != "all":
        # Generate embeddings for a specific port
        generate_embeddings_for_port(
            args.port, args.imagery_dir, args.output_dir, args.embedding_dim
        )
    else:
        # Generate embeddings for all ports
        generate_embeddings_for_all_ports(
            args.imagery_dir, args.output_dir, args.embedding_dim
        )