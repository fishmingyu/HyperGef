## HyperGef

This is a repo of Mlsys'23 paper[to apppear] HyperGef, a novel system of designing efficient fusion kernel for Hypergraph Nerual Network. Here we focus on accelerating [HGNNConv](https://ojs.aaai.org/index.php/AAAI/article/view/4235) and [UniGNNConv](https://arxiv.org/abs/2105.00956), which are classic models of hypergraph GNN. 
By kernel fusion and workload balance, our backend can bring significant speedup compared to previous SpMM backend. We compare our implementation to the cuSPARSE baseline in the kernel setting, and compare to the PyG and DGL framework in the end2end setting.

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
Note that currently pyg don't support torch 2.0+, so make sure you use pytorch 1.x version. Also, please download the optional dependencies of pyg like torch_scatter and torch_sparse libraries. 

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
cd HyperGef/data
bash prepare.sh
```

### How to Run Model
Run HGNN with our backend
```bash
cd HyperGef
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
cd HyperGef/source
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
