# Recovery Document for H3 Satellite Search

This document provides detailed instructions for recovering from failures or continuing work on the H3 Satellite Search project.

## Project Overview

The H3 Satellite Search project is designed to:
1. Generate H3 tiles (resolution 6) for specific ports (Tianjin, Dalian, Shanghai, Qingdao)
2. Download satellite imagery for each tile using Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
3. Generate embeddings from the imagery using a pre-trained ResNet model
4. Index the embeddings in a Qdrant vector database for similarity search
5. Perform similarity searches to find similar locations based on visual patterns

### Current Status

As of the last run:
- Successfully generated H3 tiles for all ports at resolution 6:
  - Tianjin: 6 tiles
  - Dalian: 13 tiles
  - Shanghai: 34 tiles
  - Qingdao: 16 tiles
- Downloaded satellite imagery for all ports using COPERNICUS/S2_SR_HARMONIZED
- Generated embeddings for all ports
- Indexed embeddings in Qdrant vector database
- Created similarity search functionality and testing scripts

## Environment Setup

### Required Dependencies

The project requires the following key dependencies:
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

### Google Earth Engine Authentication

1. Ensure the `gee_credentials.json` file is present in the root directory
2. The credentials file should contain the service account information for Google Earth Engine
3. Test authentication with:
```python
import ee
credentials = ee.ServiceAccountCredentials('strange@ee-strange.iam.gserviceaccount.com', 'gee_credentials.json')
ee.Initialize(credentials)
```

### Directory Structure

The project uses the following directory structure:
- `data/raw/` - Raw satellite imagery for each port
- `data/processed/` - Processed data including H3 tiles and vector database
- `data/embeddings/` - Generated embeddings for each tile
- `utils/` - Utility functions for each step of the pipeline
- `notebooks/` - Jupyter notebooks for analysis and visualization

## Data Processing Pipeline

The pipeline can be run for individual ports with:

```bash
python process_port.py --port tianjin --visualize
```

To run the entire pipeline for all ports:

```bash
python process_port.py --port tianjin --visualize
python process_port.py --port dalian --visualize
python process_port.py --port shanghai --visualize
python process_port.py --port qingdao --visualize
python combine_tiles.py
```

This will execute all steps of the pipeline. You can also run individual steps as described below.

### 1. Tiling Generation

The `utils/tiling.py` script generates H3 tiles for regions of interest:
- Input: Port coordinates (defined in `utils/ports.py`)
- Output: H3 tile IDs and geometries stored in `data/processed/{port_name}_tiles.geojson`

Current status:
- Successfully generated tiles for all ports at resolution 6:
  - Tianjin: 6 tiles
  - Dalian: 13 tiles
  - Shanghai: 34 tiles
  - Qingdao: 16 tiles

If tiling fails:
- Check port coordinates in `utils/ports.py`
- Consider adjusting the H3 resolution (currently set to 6)
- Verify the bounding box calculation in `get_port_bbox()` function
- Run `python utils/tiling.py --port {port_name} --resolution 6 --visualize` to regenerate tiles for a specific port

### 2. Imagery Download

The `utils/imagery.py` script downloads satellite imagery for each tile:
- Input: H3 tiles from `data/processed/{port_name}_tiles.geojson`
- Output: Satellite images stored in `data/raw/{port_name}/{tile_id}.tif`

Current status:
- Successfully downloaded imagery for all ports using COPERNICUS/S2_SR_HARMONIZED
- Using date range 2023-01-01 to 2024-12-31 with max 20% cloud cover

If imagery download fails:
- Check Google Earth Engine authentication
- Verify date range (default: 2023-01-01 to 2024-12-31) and cloud cover parameters (default: max 20%)
- Adjust the scale parameter if images are too large or too small
- Run `python utils/imagery.py --port {port_name}` to download imagery for a specific port

### 3. Embedding Generation

The `utils/embeddings.py` script generates embeddings from the imagery:
- Input: Satellite images from `data/raw/{port_name}/{tile_id}.tif`
- Output: Embeddings stored in `data/embeddings/{port_name}/{tile_id}.npy`

Current status:
- Successfully generated embeddings for all ports:
  - Tianjin: 7 embeddings
  - Dalian: 13 embeddings
  - Shanghai: 34 embeddings
  - Qingdao: 16 embeddings
- Using ResNet50 model with 2048-dimensional embeddings

If embedding generation fails:
- Check if imagery files exist and are valid
- Verify the model is loading correctly
- Adjust preprocessing parameters if needed
- Run `python utils/embeddings.py --port {port_name}` to generate embeddings for a specific port

### 4. Vector Indexing

The `utils/indexing.py` script indexes the embeddings for vector search:
- Input: Embeddings from `data/embeddings/{port_name}/{tile_id}.npy`
- Output: Qdrant vector database stored in `data/processed/vector_db`

Current status:
- Successfully indexed all embeddings (70 total) in the Qdrant vector database
- Using Cosine similarity metric
- Collection name: satellite_embeddings

If indexing fails:
- Check if embedding files exist and have the correct dimensions
- Verify Qdrant configuration (collection name, vector dimension, etc.)
- Run `python process_port.py --port {port_name} --recreate_index` to rebuild the vector index for a specific port

### 5. Similarity Search

The `notebooks/similarity_search.ipynb` notebook and `test_similarity_search.py` script demonstrate similarity search:
- Input: Query coordinates or example tile
- Output: Similar locations ranked by similarity score

Current status:
- Advanced similarity search functionality is working
- Can search by coordinates, example tile, or multiple examples
- Visualization of search results on interactive maps
- Comparison of satellite imagery for search results
- Custom search for specific infrastructure types (e.g., grain elevators)

If search fails:
- Check if vector database exists and contains indexed embeddings
- Verify query parameters (coordinates, H3 index, etc.)
- Run `python test_similarity_search.py` to test basic search functionality
- Run `python test_multi_example_search.py` to test multi-example search functionality

## Port Coordinates and H3 Resolution

Current port coordinates are defined in `utils/ports.py`:

| Port     | Latitude | Longitude |
|----------|----------|-----------|
| Tianjin  | 39.0000  | 117.7500  |
| Dalian   | 38.9500  | 121.6333  |
| Shanghai | 31.2304  | 121.4737  |
| Qingdao  | 36.0671  | 120.3826  |

If you need to modify or add port coordinates:

1. Edit `utils/ports.py` to update the coordinates
2. Adjust the H3 resolution if needed (currently set to 6)
3. Run the entire pipeline again for each port: `python process_port.py --port {port_name} --visualize`

### H3 Resolution Considerations

- Current resolution: 6 (average hexagon edge length of ~2.3 km)
- For different levels of detail, consider adjusting the resolution:
  - Resolution 4: ~20 km edge length (too coarse for most ports)
  - Resolution 5: ~8 km edge length (good for large areas)
  - Resolution 6: ~2.3 km edge length (current, good balance)
  - Resolution 7: ~1 km edge length (more detailed but generates many tiles)
- Higher resolutions will generate more tiles and require more processing time and storage

## Common Issues and Solutions

### No Tiles Generated for Some Ports

```
Generated 0 H3 cells for [port_name] at resolution 6
```

Solution:
- Try a different H3 resolution (5 or 7) depending on the port size
- Verify the port coordinates are correct
- Adjust the bounding box calculation in `get_port_bbox()` function
- Run with visualization: `python process_port.py --port {port_name} --visualize`

### Google Earth Engine Authentication Errors

```
Error authenticating with GEE: [Error message]
```

Solution:
- Verify the credentials file exists and has correct permissions
- Check if the service account has access to Earth Engine
- Ensure the service account email is correct
- Try re-authenticating manually:
  ```python
  import ee
  credentials = ee.ServiceAccountCredentials('strange@ee-strange.iam.gserviceaccount.com', 'gee_credentials.json')
  ee.Initialize(credentials)
  ```

### Missing or Low-Quality Imagery

```
No imagery found for tile [tile_id]
```

Solution:
- Extend the date range in `utils/imagery.py` (currently 2023-01-01 to 2024-12-31)
- Increase the cloud cover threshold (currently 20%)
- Try different satellite sources (currently using COPERNICUS/S2_SR_HARMONIZED)
- Adjust the scale parameter if images are too large or too small

### Embedding Generation Errors

```
Error generating embedding for [tile_id]
```

Solution:
- Check if the imagery file exists and is valid
- Verify the model is loading correctly
- Adjust preprocessing parameters in `utils/embeddings.py`
- Try a different pre-trained model if needed

### Vector Search Errors

```
Error searching vector database
```

Solution:
- Check if the vector database exists and contains indexed embeddings
- Verify the query vector has the correct dimensions (2048)
- Rebuild the vector index: `python process_port.py --port {port_name} --recreate_index`
- Check Qdrant client version compatibility

## Next Steps for Development

1. **Expand Port Coverage**:
   - Add more ports of interest globally
   - Consider adjusting resolution for specific ports based on size

2. **Enhance Embedding Quality**:
   - Experiment with different pre-trained models
   - Fine-tune models on maritime infrastructure

3. **Expand Search Capabilities**:
   - Implement filtering by port, date, or other metadata
   - Add temporal analysis to detect changes over time

4. **Improve Visualization**:
   - Create interactive maps for search results
   - Add satellite imagery overlay options

5. **Optimize Performance**:
   - Implement batch processing for large datasets
   - Add caching for frequently accessed embeddings

## Continuing Development

To continue development:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Google Earth Engine authentication with the provided credentials
4. Run the pipeline for individual ports:
   ```bash
   python process_port.py --port tianjin --visualize
   python process_port.py --port dalian --visualize
   python process_port.py --port shanghai --visualize
   python process_port.py --port qingdao --visualize
   ```
5. Combine all tiles: `python combine_tiles.py`
6. Use the test script to verify search functionality: `python test_similarity_search.py`
7. Explore the Jupyter notebook for more advanced search options

## Handling Large Files

The project generates several large files that are not suitable for version control:
- Raw satellite imagery in `data/raw/`
- Vector database in `data/processed/vector_db/`
- Embeddings in `data/embeddings/`

These directories are excluded from git using `.gitignore`. To preserve these files between Docker sessions, follow these instructions:

### Using Backup and Restore Scripts

The project includes two scripts to simplify backing up and restoring large data files:

1. **backup_data.sh**: Creates compressed archives of the data directories
2. **restore_data.sh**: Restores data from the compressed archives

To backup data:
```bash
# From inside the Docker container
cd /workspace/h3_satellite_search
./backup_data.sh
```

This will create compressed archives in `/workspace/h3_satellite_search_backup/`:
- `raw_data.tar.gz`: Raw satellite imagery
- `embeddings.tar.gz`: Generated embeddings
- `vector_db.tar.gz`: Vector database

To restore data:
```bash
# From inside the Docker container
cd /workspace/h3_satellite_search
./restore_data.sh
```

### Copying Large Files Out of Docker

To copy the backup files from the Docker container to your local machine:

```bash
# From your local machine (not inside Docker)
# Replace <container_id> with your Docker container ID
# Replace <local_destination> with the path on your local machine

# First, run the backup script inside the container
docker exec <container_id> /workspace/h3_satellite_search/backup_data.sh

# Then copy the backup directory
docker cp <container_id>:/workspace/h3_satellite_search_backup <local_destination>/h3_satellite_search_backup
```

To find your container ID, run:
```bash
docker ps
```

You can also copy the data directories directly if you prefer:

```bash
# Copy the entire data directory
docker cp <container_id>:/workspace/h3_satellite_search/data <local_destination>/h3_satellite_search_data

# Or copy specific subdirectories
docker cp <container_id>:/workspace/h3_satellite_search/data/raw <local_destination>/h3_satellite_search_data_raw
docker cp <container_id>:/workspace/h3_satellite_search/data/embeddings <local_destination>/h3_satellite_search_data_embeddings
docker cp <container_id>:/workspace/h3_satellite_search/data/processed/vector_db <local_destination>/h3_satellite_search_vector_db
```

### Copying Large Files Back into Docker

To copy the backup files back into a new Docker container:

```bash
# From your local machine (not inside Docker)
# Replace <container_id> with your new Docker container ID
# Replace <local_source> with the path on your local machine

# First, copy the backup directory to the new container
docker cp <local_source>/h3_satellite_search_backup <container_id>:/workspace/h3_satellite_search_backup

# Then run the restore script inside the container
docker exec <container_id> /workspace/h3_satellite_search/restore_data.sh
```

You can also copy the data directories directly if you prefer:

```bash
# Copy the entire data directory
docker cp <local_source>/h3_satellite_search_data <container_id>:/workspace/h3_satellite_search/data

# Or copy specific subdirectories
docker cp <local_source>/h3_satellite_search_data_raw <container_id>:/workspace/h3_satellite_search/data/raw
docker cp <local_source>/h3_satellite_search_data_embeddings <container_id>:/workspace/h3_satellite_search/data/embeddings
docker cp <local_source>/h3_satellite_search_vector_db <container_id>:/workspace/h3_satellite_search/data/processed/vector_db
```

### Using rsync for Efficient Transfers

If you have rsync available, it's more efficient for large transfers:

```bash
# Install rsync in the Docker container if needed
apt-get update && apt-get install -y rsync

# From your local machine
rsync -avz <local_source>/h3_satellite_search_data/ <user>@<docker_host>:/path/to/mount/h3_satellite_search/data/
```

### Using Mounted Volumes

For a more permanent solution, consider using Docker volumes when starting your container:

```bash
docker run -v /path/on/host/data:/workspace/h3_satellite_search/data -it <your_docker_image>
```

This will mount the `/path/on/host/data` directory from your host machine to `/workspace/h3_satellite_search/data` in the container, allowing files to persist between container restarts.

## Contact

For assistance, please contact the project maintainer.