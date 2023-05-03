import os
import sys
import subprocess
import ncu_report
import datetime

def execute_cmd(cmd):
    cmd = ' '.join(cmd)
    print(cmd)
    subprocess.call(cmd, shell=True)

def get_cf_metrics(my_action,cf_metrics):
    tmp = 0
    for m in cf_metrics:
        tmp += my_action.metric_by_name(m).as_double()
    return tmp

f = open('prof_names.txt','r')
all_metric = f.readline().split(',')
f.close()

matrices_dir = '../HyperGsys/data/mtx' #sys.argv[1]
results_dir = './profile/'
hardware = '3090'
feature_size = '32'
prof_dir = './profile'
execute = './fig8'
cuda_home = os.getenv("CUDA_HOME")
library_path = cuda_home + "/lib64"
ncu_path = cuda_home + "/bin/ncu"
ncu = f"sudo CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH={library_path} {ncu_path}"

def run_ncu():
    os.path.exists("./profile/"+hardware)
    os.system(f"mkdir -p ./profile/{hardware}")
    for (root, dirs, files) in os.walk(matrices_dir):
        print(files)
        files[:] = [f for f in files if (f.endswith(".mtx"))]
        for filename in files:
            pathmtx = os.path.join(matrices_dir, filename)
            #cmd = [ncu, '-o', results_dir+hardware+'/'+input_matrix+'-'+hardware+'-'+feature_size,'-f','--metrics',profs,'--target-processes', 'all',execute,input_matrix_dir, feature_size] # '--replay-mode','application',
            cmd = [ncu, '-o', results_dir+hardware+'/'+filename+'-'+hardware+'-'+feature_size,'-f','--set full','--target-processes all',execute, pathmtx, feature_size]
            execute_cmd(cmd)
            cmd = ['sudo','chmod','777',results_dir+hardware+'/'+filename+'-'+hardware+'-'+feature_size+'.ncu-rep']
            execute_cmd(cmd)


def analysis():
    now = datetime.datetime.now()
    os.path.exists("./profile/tables"+hardware)
    os.system(f"mkdir -p ./profile/tables/{hardware}")
    f = open(results_dir+'tables/'+hardware+'/'+now.strftime("%m%d-%H%M%S")+'-'+hardware+'.csv','w')
    launch_metric = ['launch__block_dim_x','launch__block_dim_y','launch__grid_dim_x','launch__grid_dim_y','launch__block_size', 'launch__grid_size']
    nick_names = ['L1_Red','L1_Global','L1_Hit','L2_Load','L2_Red','L2_Hit','DRAM_Read','DRAM_Write','Waves_per_SM','ActiveWarps_per_SM','Time','SOL_Mem','SOL_SM','BlockDimX','BlockDimY','GridDimX','GridDimY','BlockSize','GridSize']
    headline = ''
    all_metric.extend(launch_metric)
    for i in all_metric:
        headline += (nick_names[all_metric.index(i)]+',')
    headline += 'dataset,kernel,action'
    headline += '\n'
    f.write(headline)
    wrongf = open('wrong_names.txt','w')
    cf_metrics = ['dram__sectors_read.sum','dram__sectors_write.sum']
    cf_nick_name = 'DRAM_Read_Write'
    cf = open(results_dir+'tables/'+hardware+'/'+now.strftime("%m%d-%H%M%S")+'-'+hardware+'_'+cf_nick_name+'.csv','w')
    cf_cusparse_table = {}
    cf_ours_table = {}
    Wrong = False
    iter = 10

    for (root, dirs, files) in os.walk(matrices_dir):
        print(files)
        files[:] = [f for f in files if (f.endswith(".mtx"))]
        for filename in files:
            pathmtx = os.path.join(matrices_dir, filename)
            prof_name = results_dir+hardware+'/'+filename+'-'+hardware+'-'+feature_size+'.ncu-rep'
            print(prof_name)
            my_context = ncu_report.load_report(prof_name)
            my_range = my_context.range_by_idx(0)
            cf_ours_table[filename] = 0
            cf_cusparse_table[filename] = 0
            for j in range(my_range.num_actions()):
                my_action = my_range.action_by_idx(j)
                kernel_name = my_action.name()
                for i in all_metric:
                    if my_action.metric_by_name(i) is None:
                        Wrong = True
                        if i == 0:
                            wrongf.write(prof_name+',')
                    else:
                        f.write(str(my_action.metric_by_name(i).as_double())+',')
                if not Wrong:
                    f.write(prof_name+','+kernel_name+','+str(j))
                    f.write('\n')
                
                if j >= 30 and j < 40:
                    cf_ours_table[filename] += get_cf_metrics(my_action,cf_metrics)
                elif j < 3 * iter:
                    cf_cusparse_table[filename] += get_cf_metrics(my_action,cf_metrics)
                Wrong = False

    f.close()
    wrongf.close()

    cf_headline = 'dataset,cusparse'+',ours,ratio\n'
    cf.write(cf_headline)
    for (k,v) in cf_cusparse_table.items():
        ours_v = cf_ours_table[k]
        cf.write(k+','+str(v)+','+str(ours_v)+',' + str(ours_v/v) + '\n')
    cf.close()


if __name__ == "__main__":
    run_ncu()
    analysis()
