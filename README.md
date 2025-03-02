# Point-cloud-segmentation

## Overview
This project performs point cloud segmentation using Graph Neural Networks (GNNs) on the KITTI dataset. The pipeline includes:
- Loading and preprocessing point cloud data from `.bin` files
- Constructing a k-NN graph for point cloud representation
- Implementing a GNN-based clustering model with min-cut loss
- Visualizing segmented clusters using `pptk`

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install numpy torch torch-geometric pptk
```

## Dataset
The project uses KITTI point cloud data stored in the `./Kitti_dataset/training/velodyne/` directory. Each `.bin` file contains 3D point cloud data.

## Usage
### 1. Run the Point Cloud Segmentation
Execute the script to process the dataset, train the GNN model, and visualize clusters:

```bash
python segmentation.py
```

### 2. Modify Parameters
You can adjust model parameters like `num_clusters`, `conv_hidden`, and `mlp_hidden` in the `GNNpool` class.

## Project Structure
```
├── Kitti_dataset/
│   ├── training/
│   │   ├── velodyne/
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
├── segmentation.py   # Main script for segmentation
├── README.md         # Project documentation
```

## Future Improvements
- Experiment with different GNN architectures (e.g., GAT, GraphSAGE)
- Use real adjacency matrices instead of identity matrices for loss computation
- Optimize graph construction for large-scale datasets

## License
This project is open-source and available under the MIT License.
