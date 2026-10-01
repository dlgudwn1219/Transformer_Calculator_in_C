#!/bin/bash

# Train images download
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz

# Train labels download
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz

# Test images download
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz

# Test labels download
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

echo "Download completed... Unzipping .gz files.."

# Unzip downloaded files
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "Unzipping completed!!"
