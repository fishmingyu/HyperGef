import torch
import numpy as np

class balance_schedule:
    def __init__(self, ngs, H_T_csrptr):
        self.nrow = H_T_csrptr.shape[0] - 1
        self.work_p_sum = 0
        self.balan_key = []
        self.balan_row = []
        self.group_st = []
        self.group_ed = []
        self.ngs = ngs
        self.balancer(H_T_csrptr)
    
    def balancer(self, csrptr):
        for rid in range(self.nrow):
            A_lb = csrptr[rid]
            A_hb = csrptr[rid + 1]
            workload = int(np.ceil((A_hb - A_lb) / self.ngs))
            tmp_key = A_lb.item()
            while tmp_key < A_hb:
                self.balan_key.append(tmp_key)
                tmp_key += self.ngs
            for i in range(workload):
                for j in range(workload):
                    id1 = self.work_p_sum + i
                    id2 = self.work_p_sum + j
                    self.group_st.append(id2)
                    self.group_ed.append(id1)
                    self.balan_row.append(rid)
            self.work_p_sum += workload
        if self.balan_key[-1] != csrptr[-1]:
            self.balan_key.append(csrptr[-1].item())