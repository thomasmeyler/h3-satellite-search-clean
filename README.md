# H3 Satellite Search

[![GitHub](https://img.shields.io/github/license/thomasmeyler/h3-satellite-search-clean)](https://github.com/thomasmeyler/h3-satellite-search-clean/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

A standalone solution for satellite imagery analysis using Uber's H3 geospatial indexing system at resolution 6. This project enables:

1. Tiling regions of interest (Tianjin, Dalian, Shanghai, Qingdao ports) using H3 resolution 6
2. Downloading satellite imagery for each tile using Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
3. Generating embeddings from the imagery using a pre-trained ResNet model
4. Indexing the embeddings in a Qdrant vector database for similarity search
5. Performing similarity searches to find similar locations based on visual patterns

## Key Features

- **H3 Geospatial Indexing**: Uses Uber's H3 system for efficient spatial indexing and analysis
- **Satellite Imagery**: Downloads high-quality Sentinel-2 imagery with low cloud cover
- **Deep Learning Embeddings**: Generates 128-dimensional embeddings using a pre-trained ResNet model
- **Vector Similarity Search**: Uses Qdrant for efficient similarity search with filtering capabilities
- **End-to-End Pipeline**: Complete workflow from tiling to search in a single command

## Requirements

The project requires the following key Python packages:

- h3 (Uber's H3 geospatial indexing system)
- geopandas (for geospatial data handling)
- earthengine-api (for Google Earth Engine access)
- rasterio (for raster data processing)
- torch (for deep learning and embedding generation)
- qdrant-client (for vector similarity search)
- matplotlib, folium (for visualization)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Google Earth Engine Authentication

The project requires authentication with Google Earth Engine. You need to provide your own credentials:

1. Create a service account in the Google Cloud Console
2. Generate a private key for the service account
3. Replace the placeholder values in `gee_credentials.json` with your actual credentials
4. Alternatively, run the `save_credentials.py` script after updating it with your credentials:

```bash
# First update the credentials in save_credentials.py
python save_credentials.py
```

5. Test authentication with:

```python
import ee
with open('gee_credentials.json', 'r') as f:
    creds = json.load(f)
credentials = ee.ServiceAccountCredentials(creds['client_email'], 'gee_credentials.json')
ee.Initialize(credentials)
```

**Note:** For security reasons, never commit your actual Google Earth Engine credentials to version control. The repository contains placeholder values that need to be replaced with your own credentials.

## Project Structure

```
h3_satellite_search/
├── data/
│   ├── raw/             # Raw satellite imagery for each port
│   ├── processed/       # Processed data (H3 tiles, vector database)
│   └── embeddings/      # Generated embeddings for each tile
├── notebooks/           # Jupyter notebooks for analysis and visualization
├── utils/               # Utility functions for each step of the pipeline
│   ├── ports.py         # Port coordinates and bounding box functions
│   ├── tiling.py        # H3 tiling functions
│   ├── imagery.py       # Satellite imagery download functions
│   ├── embeddings.py    # Embedding generation functions
│   ├── indexing.py      # Vector database indexing functions
│   └── visualization.py # Visualization functions
├── main.py              # Main script to run the entire pipeline
├── test_similarity_search.py # Test script for similarity search
├── README.md            # Project documentation
├── RECOVERY.md          # Recovery procedures and troubleshooting
├── gee_credentials.json # Google Earth Engine credentials
└── requirements.txt     # Python dependencies
```

## Usage

### Running the Pipeline for Individual Ports

To run the pipeline for a single port:

```bash
python process_port.py --port tianjin --visualize
```

This will:
1. Generate H3 tiles for the specified port
2. Download satellite imagery for each tile
3. Generate embeddings for each image
4. Index the embeddings in the vector database

### Running the Full Pipeline

To run the entire pipeline for all ports:

```bash
python process_port.py --port tianjin --visualize
python process_port.py --port dalian --visualize
python process_port.py --port shanghai --visualize
python process_port.py --port qingdao --visualize
python combine_tiles.py
```

### Testing Similarity Search

To test the similarity search functionality:

```bash
python test_similarity_search.py
```

### Using the Jupyter Notebook

For more advanced search options and visualization:

```bash
jupyter notebook notebooks/similarity_search.ipynb
```

The notebook provides several search methods:

1. **Search by Coordinates**: Find similar locations based on specific coordinates
2. **Search by Example Tile**: Use an existing tile as a reference to find similar areas
3. **Search by Multiple Examples**: Combine multiple examples to find specific patterns (e.g., grain elevators)
4. **Visualize Results**: View search results on interactive maps and compare satellite imagery

## Ports of Interest

The project focuses on the following ports:

| Port     | Latitude | Longitude | Description                       |
|----------|----------|-----------|-----------------------------------|
| Tianjin  | 39.0000  | 117.7500  | Major port in Northern China      |
| Dalian   | 38.9500  | 121.6333  | Important port in Northeast China |
| Shanghai | 31.2304  | 121.4737  | Busiest container port in the world |
| Qingdao  | 36.0671  | 120.3826  | Major port in Eastern China       |

## Current Status

- Successfully generated H3 tiles for all ports at resolution 6:
  - Tianjin: 6 tiles
  - Dalian: 13 tiles
  - Shanghai: 34 tiles
  - Qingdao: 16 tiles
- Downloaded satellite imagery for all ports using COPERNICUS/S2_SR_HARMONIZED
- Generated embeddings for all ports
- Indexed embeddings in Qdrant vector database
- Created similarity search functionality and testing scripts

## Handling Large Files

The project generates several large files that are not suitable for version control:
- Raw satellite imagery in `data/raw/`
- Vector database in `data/processed/vector_db/`
- Embeddings in `data/embeddings/`

To backup and restore these files between Docker sessions, use the provided scripts:

```bash
# To backup data
./backup_data.sh

# To restore data
./restore_data.sh
```

For detailed instructions on copying files in and out of Docker containers, refer to the `RECOVERY.md` document.

## Recovery Document

For detailed troubleshooting steps, recovery procedures, and next steps for development, refer to the `RECOVERY.md` document.