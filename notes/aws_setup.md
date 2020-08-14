aws ec2 r6g.medium

Ubuntu 18.04

```
sudo apt update -y

# pyenv python
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# opencv
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev  libdc1394-22-dev
# libjasper-dev

#scipy
sudo apt install libblas3 liblapack3 liblapack-dev libblas-dev
sudo apt install gfortran
sudo apt install libatlas-base-dev

# plyvel
sudo apt install libleveldb-dev

# bazel
sudo apt install g++ unzip zip
sudo apt-get install openjdk-11-jdk

sudo apt install libhdf5-dev

sudo apt install python3-dev

curl https://pyenv.run | bash
nano ~/.bashrc
#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"
pyenv install 3.8.5
pyenv global 3.8.5

python -m pip install keras_applications --no-deps
python -m pip install keras_preprocessing --no-deps
python -m pip install pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1'
python -m pip install absl-py
python -m pip install h5py

wget https://github.com/bazelbuild/bazel/releases/download/3.4.1/bazel-3.4.1-linux-arm64
chmod +x bazel-3.4.1-linux-arm64
mkdir /home/ubuntu/bin
mv bazel-3.4.1-linux-arm64 $HOME/bin/bazel
nano ~/.bashrc 
# export PATH="$PATH:$HOME/bin"

#tensorflow
#https://www.tensorflow.org/install/source#install_bazel
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.3
./configure
bazel build //tensorflow/tools/pip_package:build_pip_package --config=noaws
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
mv /tmp/tensorflow_pkg/tensorflow-2.3.0-cp38-cp38-linux_aarch64.whl ~/

python -m pip install ~/tensorflow-2.3.0-cp38-cp38-linux_aarch64.whl

# python -m pip install scipy
python -m pip install matplotlib
python -m pip install jupyter
python -m pip install scikit-build
python -m pip install opencv-python
python -m pip install hnswlib
python -m pip install networkx
python -m pip install plyvel

curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

```