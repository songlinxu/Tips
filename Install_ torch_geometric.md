## Step 1: create conda virtual environment

conda create -n pygdemo python=3.8
conda activate pygdemo

## Step 2: 

Download torch-1.8.0+cu101-cp38-cp38-linux_x86_64.whl from https://download.pytorch.org/whl/torch_stable.html

## Step 3: 

Go into the folder contains thiss whl file and install: pip install torch-1.8.0+cu101-cp38-cp38-linux_x86_64.whl


## Step 4: 

pip install --no-index torch_scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install --no-index torch_sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install --no-index torch_cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install --no-index torch_spline_conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

## Step 5:

pip install torch_geometric





