import os
data_name = ["yelp", "20newsW100", "coauthor_cora", "zoo", "NTU2012", "cora", "pubmed", "Mushroom", 
                  "coauthor_dblp", "house-committees", "walmart-trips", "citeseer", "ModelNet40"]
feature_list = [32, 64, 128]
backend_list = ['pyg', 'dgl', 'hgsys']
model_list = ["HGNN", "UniGIN", "UniGCNII"]

for dname in data_name:
    for nhid in feature_list:
        for backend in backend_list:
            for model in model_list:
                cmd = f"python ../HyperGsys/hgsys.py --dname {dname} --nhid {nhid} --backend {backend} --model {model} --data-path ../HyperGsys/data --output fig6.csv"
                print(cmd)
                os.system(cmd)

            