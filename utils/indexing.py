"""
Vector indexing utilities for satellite imagery embeddings.

This module provides functions for indexing embeddings in a vector database.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from utils.ports import get_all_ports

# Default indexing parameters
DEFAULT_COLLECTION_NAME = "satellite_embeddings"
DEFAULT_DISTANCE = "Cosine"  # Cosine, Euclid, Dot

def connect_to_qdrant(persist_path=None):
    """
    Connect to a Qdrant vector database.
    
    Args:
        persist_path: Path to the database (None for in-memory)
        
    Returns:
        QdrantClient
    """
    # Create Qdrant client
    if persist_path:
        os.makedirs(persist_path, exist_ok=True)
        client = QdrantClient(path=persist_path)
    else:
        client = QdrantClient(":memory:")
    
    return client

def setup_vector_db(collection_name=DEFAULT_COLLECTION_NAME, vector_dim=None, distance=DEFAULT_DISTANCE, persist_path=None):
    """
    Set up a Qdrant vector database.
    
    Args:
        collection_name: Name of the collection
        vector_dim: Dimension of the vectors
        distance: Distance metric (Cosine, Euclid, Dot)
        persist_path: Path to persist the database (None for in-memory)
        
    Returns:
        QdrantClient
    """
    # Create Qdrant client
    client = connect_to_qdrant(persist_path)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection {collection_name} already exists")
    else:
        # Create collection
        if vector_dim is None:
            raise ValueError("vector_dim must be specified when creating a new collection")
        
        distance_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Euclid": qmodels.Distance.EUCLID,
            "Dot": qmodels.Distance.DOT
        }
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_dim,
                distance=distance_map.get(distance, qmodels.Distance.COSINE)
            )
        )
        print(f"Created collection {collection_name} with vector dimension {vector_dim} and distance metric {distance}")
    
    return client

def load_embeddings_for_port(port_name, embeddings_dir):
    """
    Load embeddings for a specific port.
    
    Args:
        port_name: Name of the port
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Dictionary mapping H3 indices to embeddings
    """
    port_embeddings_dir = os.path.join(embeddings_dir, port_name)
    
    if not os.path.exists(port_embeddings_dir):
        print(f"No embeddings found for port: {port_name}")
        return {}
    
    # Load metadata
    metadata_path = os.path.join(port_embeddings_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        h3_indices = metadata.get('h3_indices', [])
    else:
        # Get all embedding files
        embedding_files = [f for f in os.listdir(port_embeddings_dir) if f.endswith('.npy')]
        h3_indices = [os.path.splitext(f)[0] for f in embedding_files]
    
    # Load embeddings
    embeddings = {}
    
    for h3_index in h3_indices:
        embedding_path = os.path.join(port_embeddings_dir, f"{h3_index}.npy")
        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path)
            embeddings[h3_index] = embedding
    
    print(f"Loaded {len(embeddings)} embeddings for {port_name}")
    return embeddings

def load_embeddings_for_all_ports(embeddings_dir):
    """
    Load embeddings for all ports.
    
    Args:
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Dictionary mapping port names to embeddings
    """
    all_embeddings = {}
    
    for port_name in get_all_ports():
        port_embeddings = load_embeddings_for_port(port_name, embeddings_dir)
        all_embeddings[port_name] = port_embeddings
    
    return all_embeddings

def load_tiles_gdf(tiles_path):
    """
    Load tiles GeoDataFrame.
    
    Args:
        tiles_path: Path to tiles GeoJSON file
        
    Returns:
        GeoDataFrame
    """
    return gpd.read_file(tiles_path)

def index_embeddings(client, port_name=None, embeddings_dir=None, collection_name=DEFAULT_COLLECTION_NAME, tiles_gdf=None, recreate=False):
    """
    Index embeddings in the vector database.
    
    Args:
        client: QdrantClient
        port_name: Name of the port (if None, index all ports)
        embeddings_dir: Directory containing embeddings
        collection_name: Name of the collection
        tiles_gdf: GeoDataFrame with tile geometries
        recreate: Whether to recreate the collection
        
    Returns:
        None
    """
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    # If recreate is True, delete the collection if it exists
    if recreate and collection_name in collection_names:
        client.delete_collection(collection_name)
        print(f"Deleted collection {collection_name}")
        collection_names.remove(collection_name)
    
    # Load embeddings
    if port_name is not None:
        # Load embeddings for a single port
        port_embeddings = load_embeddings_for_port(port_name, embeddings_dir)
        embeddings = {port_name: port_embeddings}
    else:
        # Load embeddings for all ports
        embeddings = load_embeddings_for_all_ports(embeddings_dir)
    
    # Get embedding dimension from the first embedding
    embedding_dim = None
    for port_embeddings in embeddings.values():
        if port_embeddings:
            embedding_dim = next(iter(port_embeddings.values())).shape[0]
            break
    
    if embedding_dim is None:
        print("No embeddings found")
        return
    
    # Create collection if it doesn't exist
    if collection_name not in collection_names:
        distance_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Euclid": qmodels.Distance.EUCLID,
            "Dot": qmodels.Distance.DOT
        }
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dim,
                distance=distance_map.get(DEFAULT_DISTANCE, qmodels.Distance.COSINE)
            )
        )
        print(f"Created collection {collection_name} with vector dimension {embedding_dim} and distance metric {DEFAULT_DISTANCE}")
    
    # Load tiles if not provided
    if tiles_gdf is None and os.path.exists("../data/processed/all_tiles.geojson"):
        tiles_gdf = load_tiles_gdf("../data/processed/all_tiles.geojson")
    
    # Prepare points for indexing
    points = []
    
    # Get the current count of points in the collection
    collection_info = client.get_collection(collection_name)
    point_id = collection_info.points_count
    
    for port_name, port_embeddings in embeddings.items():
        for h3_index, embedding in tqdm(port_embeddings.items(), desc=f"Preparing {port_name} embeddings"):
            # Get tile geometry if available
            if tiles_gdf is not None:
                tile_row = tiles_gdf[tiles_gdf['h3_index'] == h3_index]
                if len(tile_row) > 0:
                    centroid = tile_row.iloc[0]['geometry'].centroid
                    centroid_coords = [centroid.y, centroid.x]  # [lat, lng]
                else:
                    centroid_coords = None
            else:
                centroid_coords = None
            
            # Create payload
            payload = {
                'h3_index': h3_index,
                'port': port_name
            }
            
            if centroid_coords:
                payload['centroid'] = centroid_coords
            
            # Create point
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            ))
            
            point_id += 1
    
    # Upload in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
    
    print(f"Indexed {len(points)} embeddings in collection {collection_name}")

def search_similar_tiles(client, collection_name, query_embedding, top_k=5, filter_condition=None):
    """
    Search for similar tiles in the vector database.
    
    Args:
        client: QdrantClient
        collection_name: Name of the collection
        query_embedding: Query embedding
        top_k: Number of results to return
        filter_condition: Filter condition
        
    Returns:
        List of search results
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        query_filter=filter_condition
    )
    
    return results

def create_filter_by_port(port_name):
    """
    Create a filter condition for a specific port.
    
    Args:
        port_name: Name of the port
        
    Returns:
        Filter condition
    """
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="port",
                match=qmodels.MatchValue(value=port_name)
            )
        ]
    )

def create_filter_by_location(lat, lng, radius_km):
    """
    Create a filter condition for a specific location.
    
    Args:
        lat: Latitude
        lng: Longitude
        radius_km: Radius in kilometers
        
    Returns:
        Filter condition
    """
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="centroid",
                geo_radius=qmodels.GeoRadius(
                    center=qmodels.GeoPoint(
                        lat=lat,
                        lon=lng
                    ),
                    radius=radius_km * 1000  # Convert to meters
                )
            )
        ]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index embeddings in a vector database")
    parser.add_argument("--embeddings_dir", type=str, default="../data/embeddings", help="Directory containing embeddings")
    parser.add_argument("--tiles_path", type=str, default="../data/processed/all_tiles.geojson", help="Path to tiles GeoJSON file")
    parser.add_argument("--collection_name", type=str, default=DEFAULT_COLLECTION_NAME, help=f"Collection name (default: {DEFAULT_COLLECTION_NAME})")
    parser.add_argument("--distance", type=str, default=DEFAULT_DISTANCE, help=f"Distance metric (Cosine, Euclid, Dot) (default: {DEFAULT_DISTANCE})")
    parser.add_argument("--persist_path", type=str, default="../data/processed/vector_db", help="Path to persist the database")
    
    args = parser.parse_args()
    
    # Load embeddings
    all_embeddings = load_embeddings_for_all_ports(args.embeddings_dir)
    
    # Get embedding dimension from the first embedding
    embedding_dim = None
    for port_embeddings in all_embeddings.values():
        if port_embeddings:
            embedding_dim = next(iter(port_embeddings.values())).shape[0]
            break
    
    if embedding_dim is None:
        print("No embeddings found")
        exit(1)
    
    # Load tiles
    tiles_gdf = None
    if os.path.exists(args.tiles_path):
        tiles_gdf = load_tiles_gdf(args.tiles_path)
    
    # Set up vector database
    client = setup_vector_db(args.collection_name, embedding_dim, args.distance, args.persist_path)
    
    # Index embeddings
    index_embeddings(client, args.collection_name, all_embeddings, tiles_gdf)