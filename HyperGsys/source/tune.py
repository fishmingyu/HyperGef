import os
path = "../data/mtx"  # SuiteSparse path
for (root, dirs, files) in os.walk(path):
    print(files)
    files[:] = [f for f in files if (f.endswith(".mtx"))]
    for filename in files:
        pathmtx = os.path.join(path, filename)
        cmd = "./aggr_proto %s %d" % (pathmtx, 32)
        print(cmd)
        os.system(cmd)
        # print(cmd)