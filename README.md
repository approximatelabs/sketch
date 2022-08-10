# sketch

-- Currently in active development --

A package for simplifying the use of data-sketches in python workflows.

Includes a self-hostable fastAPI web-server for hosting sketches and hosting widgets for embedding the charts.


## Set up faiss to be available to import


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
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" stuff
# Also, can get the python path by creating a new file
python -c 'import site; print(site.getsitepackages())'
# Take note of the path above
echo "$(ls -d ./build/faiss/python/build/lib*/)" > "{{PATHHERE}}/faiss.pth"
```