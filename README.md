## HyperGNNSys

This is a repo demo for a novel system for hyper graph nerual network. Here we focus on accelerating HGNNConv, a classic model for hypergraph GNN. 
By kernel fusion and workload balance, our backend can bring significant speedup compared to SpMM backend. 

### Requirement
```
cuda >= 11.6
torch >= 1.11
dgl >= 0.9
pyg >= 2.1
pandas
sklearn
GPUtil
```

### Setup
Please first config your ```$CUDA_HOME``` environment. It probably locates at ```/usr/local/cuda``` or something like that.
Then config the ncu_report environment:
```export PYTHONPATH="${CUDA_HOME}/nsight-compute-xxxx.x.x/extras/python"```

Build the project
```bash
bash build.sh
```
Download dataset
```bash
cd HyperGsys/data
bash prepare.sh
```

### How to Run Model
Run HGNN with our backend
```bash
cd HyperGsys
```
```python
python ugsys.py --backend hgsys --model-name HGNN
```
Compared with other backend (DGL, PyG)
```python
python ugsys.py --backend dgl --model-name HGNN
python ugsys.py --backend pyg --model-name HGNN
```

### Run C++ Hyper Aggregation
```bash
cd HyperGsys/source
make
./aggr_proto ../data/mtx_data/cora.mtx [feature_length,e.g: 32]
```

### To verify functionality
```
cd test
pytest
```

### To run Artifact Evaluation
```
cd experiment
make
python fig6.py
python fig7.py
python fig8.py
python fig9.py
python fig10.py
```
The result of fig8 will appear in profile/tables directory. Other result will be presented in figX.csv file.
