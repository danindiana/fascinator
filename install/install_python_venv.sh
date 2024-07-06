#!/bin/bash

# Update package list and upgrade all packages to their latest versions
sudo apt update && sudo apt upgrade -y

# Install dependencies for building Python
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev

# Download Python 3.12.4 source code
wget https://www.python.org/ftp/python/3.12.4/Python-3.12.4.tgz

# Extract the downloaded file
tar -xf Python-3.12.4.tgz

# Navigate into the extracted directory
cd Python-3.12.4

# Configure the build
./configure --enable-optimizations

# Build and install Python 3.12.4
make -j $(nproc)
sudo make altinstall

# Verify Python 3.12.4 installation
python3.12 --version

# Create a virtual environment
python3.12 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install necessary dependencies
# Replace the following line with the actual dependencies for your project
pip install numpy pandas scikit-learn

echo "Python 3.12.4 installation, virtual environment setup, and dependency installation complete."
