import os
path = "../HyperGsys/data/mtx"  # SuiteSparse path
selected = ['Mushroom.mtx', '20newsW100.mtx']
for (root, dirs, files) in os.walk(path):
    files[:] = [f for f in files if (f.endswith(".mtx"))]
    for filename in files:
        get_split = filename.split("/")[-1]
        if get_split in selected:
            pathmtx = os.path.join(path, filename)
            cmd = "./fig10 %s %d" % (pathmtx, 32)
            print(cmd)
            os.system(cmd)
        # print(cmd)