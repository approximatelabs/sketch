

## Some setup stuff (saving things i've done)

### TODO: put this into Dockerfile and build an image

```
curl https://pyenv.run | bash
```
copy the things needed into the `.bashrc`

```
sudo apt-get install build-essential zlib1g-dev libomp-dev swig libblas-dev liblapack-dev make libssl-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev cmake
```

```
pyenv install 3.10.5
pyenv virtualenv create sketch
pyenv global sketch
```

## Setup datasketches

### for others, 
```
pip install datasketches
```

### https://github.com/apache/datasketches-cpp

for on OSX, need to do a manual build and install, since `pip install` installs the wrong architecture versions. 

```
git clone ...
make commands
# ensure right environment
python3 -m pip install .
```

```
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release
```



## Setup FAISS


(Oh, needed to... install a more recent version of cmake than apt-get was offering...)

### Cmake stuff...
```
sudo apt remove cmake #Uninstall old version

sudo apt-get install build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz

tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
./bootstrap
make
sudo make install
```
 note, I had to move the cmake binaries to `/usr/bin` (or just reference it directly i guess)

```
git clone git@github.com:facebookresearch/faiss.git
cd faiss
git fetch --all --tags
git checkout tags/v1.7.2
cmake -B build . -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=$(which python)
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
echo "$(ls -d ./build/faiss/python/build/lib*/)" > "$(python -c 'import site; print(site.getsitepackages()[0])')/faiss.pth"
```

## setup stuff needed

```
pip install -r dev-requirements.txt
```

## Next, for live-editability of css... need node and stuff

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install --lts
cd ~/sketch/sketch/api/tailwindcss && npm install
```


## mac version

```
git clone git@github.com:facebookresearch/faiss.git
cd faiss
git fetch --all --tags
git checkout tags/v1.7.2
brew install libomp swig openblas lapack
cmake -B build . -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DPython_EXECUTABLE=$(which python)
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
# make -C build install
# PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)"
# Also, can get the python path by creating a new file
python -c 'import site; print(site.getsitepackages())'
# Take note of the path above
echo "$(ls -d ./build/faiss/python/build/lib*/)" > "{{PATHHERE}}/faiss.pth"
```



# For now, hosting with ... nginx proxy manager i guess?
(really suggests i should move over to docker for everything..)

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
 ```