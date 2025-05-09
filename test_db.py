#!/usr/bin/env python3
"""
Test script to check the vector database.
"""

import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def main():
    """
    Test the vector database.
    """
    print("Testing vector database...")
    
    # Connect to vector database
    vector_db_path = "./data/processed/vector_db"
    client = QdrantClient(path=vector_db_path)
    print(f"Connected to vector database at {vector_db_path}")
    
    # Get collection info
    collection_info = client.get_collection("satellite_embeddings")
    print(f"Collection: satellite_embeddings")
    print(f"Vector dimension: {collection_info.config.params.vectors.size}")
    print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    print(f"Number of points: {collection_info.points_count}")
    
    # Load a random embedding from the tianjin directory
    embeddings_dir = "./data/embeddings"
    port_name = "tianjin"
    port_embeddings_dir = os.path.join(embeddings_dir, port_name)
    
    # Get all embedding files
    embedding_files = [f for f in os.listdir(port_embeddings_dir) if f.endswith('.npy')]
    
    if not embedding_files:
        print(f"No embeddings found for port: {port_name}")
        return
    
    # Load a random embedding
    h3_index = os.path.splitext(embedding_files[0])[0]
    embedding_path = os.path.join(port_embeddings_dir, f"{h3_index}.npy")
    embedding = np.load(embedding_path)
    print(f"Loaded embedding for {port_name} - {h3_index}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Search for the embedding in the database
    results = client.search(
        collection_name="satellite_embeddings",
        query_vector=embedding.tolist(),
        limit=5
    )
    
    print(f"\nSearch results for {h3_index}:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result.payload['port']} - {result.payload['h3_index']} (Score: {result.score:.4f})")
    
    # Try to search for a specific H3 index
    print("\nSearching for specific H3 index...")
    results = client.search(
        collection_name="satellite_embeddings",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="h3_index",
                    match=MatchValue(value=h3_index)
                )
            ]
        ),
        query_vector=embedding.tolist(),
        limit=1
    )
    
    if results:
        print(f"Found {h3_index} in the database")
    else:
        print(f"Could not find {h3_index} in the database")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()