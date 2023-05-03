import os
path = "../HyperGsys/data/mtx"  # SuiteSparse path
partition_dict = {"yelp.mtx":400, "20newsW100.mtx":400, "coauthor_cora.mtx":10, "zoo.mtx":20, "NTU2012.mtx":80, "cora.mtx":210, "pubmed.mtx":40, "Mushroom.mtx":250, 
                  "coauthor_dblp.mtx":80, "house-committees.mtx":40, "walmart-trips.mtx":210, "citeseer.mtx":6, "ModelNet40.mtx":300}
for (root, dirs, files) in os.walk(path):
    print(files)
    files[:] = [f for f in files if (f.endswith(".mtx"))]
    for filename in files:
        pathmtx = os.path.join(path, filename)
        partition = partition_dict[filename]
        cmd1 = "./fig7 %s %d %d" % (pathmtx, 32, partition)
        cmd2 = "./fig7 %s %d %d" % (pathmtx, 64, partition)
        print(cmd1)
        os.system(cmd1)
        print(cmd2)
        os.system(cmd2)
    
        # print(cmd)