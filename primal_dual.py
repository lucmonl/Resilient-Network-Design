"Using primal-dual method."

import argparse
from gurobipy import *
import numpy as np
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
import random
import multiprocessing
import multiprocessing.sharedctypes
import ctypes
random.seed(10)
np.random.seed(0)
import scipy
from scipy import sparse

from tqdm.notebook import tqdm, trange

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

path = os.path.abspath(os.path.dirname(__file__))
#type = sys.getfilesystemencoding()
file_name = os.path.basename(sys.argv[0])[:-3]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="USairport_2010", type=str, help="USairport_2010")
parser.add_argument("--perturb_num", default=12, type=int, help="# of perturbed nodes")
parser.add_argument("--budget", default=2, type=int, help="budget count")
parser.add_argument("--perturb_hub_node", default=False, type=bool, help="True: perturb hub nodes")
parser.add_argument("--coef_u", default=5, type=float, help="coefficient of standard capacity")
parser.add_argument("--coef_c", default=0.1, type=float, help="coefficient of transmission cost")
parser.add_argument("--dataset_id", default=0, type=int, help="dataset id 0~50")
parser.add_argument("--rho", default=1, type=float, help="rho")
parser.add_argument("--beta", default=100, type=float, help="beta")
parser.add_argument("--alpha", default=50, type=float, help="alpha")
parser.add_argument("--iter_num", default=1000, type=int, help="iter num")
parser.add_argument("--decay_step", default=False, type=bool, help="decay step")
args = parser.parse_args()
dataset =  args.dataset #"USairport_2010"

#print(args)
# adjacency
if dataset == "synthetic_uniform" or dataset == "power_law_53":
    n = 53 #64
    perturb_num = args.perturb_num
    intact_num = n - perturb_num
    #intact_num = 24
    budget = args.budget #0 #24
    
    a = sparse.load_npz("network/{}/a_{}.npz".format(dataset, args.dataset_id)).toarray()
    n = a.shape[0]
    print("edge count:", np.sum(a) // 2)

    y = np.ones(n)

    load_dataset = "USairport053"
    transmit_cost = np.load("problem_set/{}/{}/transmit_cost.npy".format(load_dataset,"53_41_2_0.3")) * args.coef_c
    d = np.load("problem_set/{}/{}/d.npy".format(load_dataset,"53_41_2_0.3"))
    u = np.load("problem_set/{}/{}/u.npy".format(load_dataset,"53_41_2_0.3")) * args.coef_u

    if args.perturb_hub_node:
        perturb_index = np.load("network/{}/perturb_node_hub_12_{}.npy".format(dataset, args.dataset_id))
    else:
        perturb_index = np.random.choice(n, perturb_num, replace=False)
    print("Perturbed Nodes: ", perturb_index)
    y[perturb_index] = 0
elif dataset in ["USairport300","USairport147","USairport053"]:
    n = int(dataset[-3:])
    perturb_num = 12
    intact_num = n - perturb_num
    budget = 2
    a = np.zeros((n,n))
    with open("network/{}.txt".format(dataset),"r") as txt_file:
        line = txt_file.readline()
        while line:
            arr = line.split()
            if int(arr[0]) - 1<n and int(arr[1])-1 < n:
                a[int(arr[0])-1, int(arr[1])-1] = 1
            line = txt_file.readline()
    y = np.ones(n)
    perturb_index = np.random.choice(n, perturb_num, replace=False)
    print("Perturbed Nodes: ", perturb_index)
    y[perturb_index] = 0
elif dataset in ["USairport_2010","USairport_2010_66","USairport_2010_34","USairport_2010_119","USairport_2010_850"]:
    a = np.load("network/{}/a.npy".format(args.dataset))
    n = a.shape[0]
    print("node count:", n)
    budget = args.budget
    transmit_cost = np.load("network/{}/transmit_cost.npy".format(args.dataset)) * args.coef_c
    d = np.load("network/{}/d.npy".format(args.dataset))
    u = np.load("network/{}/u.npy".format(args.dataset)) * args.coef_u
    y = np.ones(n)
    if args.perturb_hub_node:
        #perturb_index = [12, 17, 22, 35, 39,  6, 30, 38,  5, 41, 11,  9]
        perturb_index = np.load("network/{}/perturb_hub_index.npy".format(args.dataset))
        perturb_num = perturb_index.shape[0]
    else:
        perturb_num = args.perturb_num
        perturb_index = np.random.choice(n, perturb_num, replace=False)
    intact_num = n - perturb_num
    print("Perturbed Nodes: ", perturb_index)
    y[perturb_index] = 0


# edge
edge_cnt = int(np.sum(a) // 2)
edge_dict = {}
row, col = np.where(a != 0)
for i in range(row.shape[0]):
    if row[i] in edge_dict:
        edge_dict[row[i]].append(col[i])
    else:
        edge_dict[row[i]] = [col[i]]

#map (row, col) -> var index
edge_var = {}
var_index = 0
for row in edge_dict:
    for col in edge_dict[row]:
        if row < col:
            edge_var[(row, col)] = var_index
            var_index += 1
        else:
            edge_var[(row, col)] = edge_var[(col, row)]
assert edge_cnt == var_index

reduced_transmit_cost = []
for row in edge_dict:
    for col in edge_dict[row]:
        if row < col:
            reduced_transmit_cost.append(transmit_cost[row * n + col])
assert len(reduced_transmit_cost) == edge_cnt


sample_dir = "{}/{}/{}/{}/{}".format(n,intact_num,budget,args.coef_u,args.coef_c)
arg_dir = "{}_rho{}_beta_{}_alpha_{}_iter_{}_decay_{}".format(file_name, args.rho, args.beta * args.rho, args.alpha, args.iter_num, args.decay_step)
os.makedirs("problem_set/{}/{}".format(dataset, sample_dir), exist_ok=True)
os.makedirs("results/{}/{}".format(dataset, sample_dir), exist_ok=True)
sys.stdout = Logger(filename = "results/{}/{}/{}.txt".format(dataset, sample_dir, arg_dir))
print(args)


def get_eq_constrs(n):
    n_variables = int(2*edge_cnt + 2*n)
    n_eq_constrs = n

    eq_constr_coefs_row = []
    eq_constr_coefs_col = []
    eq_constr_coefs_data = []
    eq_constr_rhs = []

    constr_id = 0
    for row in range(n):
        for col in edge_dict[row]:
            eq_constr_coefs_row.append(constr_id)
            eq_constr_coefs_col.append(edge_var[(row,col)])
            if(row<col):
                eq_constr_coefs_data.append(1)
            else:
                eq_constr_coefs_data.append(-1)

        eq_constr_coefs_row.append(constr_id)
        eq_constr_coefs_col.append(2*edge_cnt+2*row)
        eq_constr_coefs_data.append(-1)

        eq_constr_coefs_row.append(constr_id)
        eq_constr_coefs_col.append(2*edge_cnt+2*row+1)
        eq_constr_coefs_data.append(1)

        eq_constr_rhs.append(-d[row])
        constr_id += 1
    assert constr_id == int(n_eq_constrs)
    eq_constr_coefs = sparse.csr_matrix((eq_constr_coefs_data, (eq_constr_coefs_row, eq_constr_coefs_col)),
                                        shape = (int(n_eq_constrs), n_variables))
    return n_eq_constrs, eq_constr_coefs, np.array(eq_constr_rhs)

def get_eq_constrs_full(n):
    n_variables = 2*n**2 + 2*n
    n_eq_constrs = n*(n+1)/2 + n*(n-1)/2 + n
    assert n_eq_constrs == int(n_eq_constrs)
    #eq_constr_coefs = np.zeros((int(n_eq_constrs), n_variables))
    eq_constr_rhs = np.zeros(int(n_eq_constrs))

    eq_constr_coefs_row = []
    eq_constr_coefs_col = []
    eq_constr_coefs_data = []
    index_id = 0
    constr_id = 0

    #x_ij == -x_ji
    for i in range(n):
        for j in range(i,n):
            eq_constr_coefs_row.append(constr_id)
            eq_constr_coefs_col.append(i*n+j)
            eq_constr_coefs_data.append(1)
            index_id += 1

            if i!=j:
                eq_constr_coefs_row.append(constr_id)
                eq_constr_coefs_col.append(j*n+i)
                eq_constr_coefs_data.append(1)
                index_id += 1

            eq_constr_rhs[constr_id] = 0
            constr_id += 1
    
    #p_ij == p_ji
    for i in range(n):
        for j in range(i+1,n):
            eq_constr_coefs_row.append(constr_id)
            eq_constr_coefs_col.append(n**2+i*n+j)
            eq_constr_coefs_data.append(1)
            index_id += 1
            
            eq_constr_coefs_row.append(constr_id)
            eq_constr_coefs_col.append(n**2+j*n+i)
            eq_constr_coefs_data.append(-1)
            index_id += 1
            eq_constr_rhs[constr_id] = 0
            constr_id += 1
    
    #sum(s_ij) - delta_neg + delta_pos == -d_i
    for i in range(n):
        for j in range(i*n, (i+1)*n):
            eq_constr_coefs_row.append(constr_id)
            eq_constr_coefs_col.append(j)
            eq_constr_coefs_data.append(1)
            index_id += 1

        eq_constr_coefs_row.append(constr_id)
        eq_constr_coefs_col.append(2*n**2+2*i)
        eq_constr_coefs_data.append(-1)
        index_id += 1

        eq_constr_coefs_row.append(constr_id)
        eq_constr_coefs_col.append(2*n**2+2*i+1)
        eq_constr_coefs_data.append(1)
        index_id += 1

        eq_constr_rhs[constr_id] = -d[i]
        constr_id += 1

    eq_constr_coefs = sparse.csr_matrix((eq_constr_coefs_data, (eq_constr_coefs_row, eq_constr_coefs_col)),
                                        shape = (int(n_eq_constrs), n_variables))
    """
    dense matrix
    #x_ij == -x_ji
    for i in range(n):
        for j in range(i,n):
            eq_constr_coefs[constr_id, i*n+j] = 1
            eq_constr_coefs[constr_id, j*n+i] = 1
            eq_constr_rhs[constr_id] = 0
            constr_id += 1
    #p_ij == p_ji
    for i in range(n):
        for j in range(i+1,n):
            eq_constr_coefs[constr_id, n**2+i*n+j] = 1
            eq_constr_coefs[constr_id, n**2+j*n+i] = -1
            eq_constr_rhs[constr_id] = 0
            constr_id += 1

    #sum(s_ij) - delta_neg + delta_pos == -d_i
    for i in range(n):
        eq_constr_coefs[constr_id, i*n:(i+1)*n] = 1
        eq_constr_coefs[constr_id, 2*n**2+2*i] = -1
        eq_constr_coefs[constr_id, 2*n**2+2*i+1] = 1
        eq_constr_rhs[constr_id] = -d[i]
        constr_id += 1
    """
    return n_eq_constrs, eq_constr_coefs, eq_constr_rhs

def get_neq_constrs(n, delta_y):
    n_variables = int(2*edge_cnt + 2*n)
    n_neq_constrs = 4 * edge_cnt

    neq_constr_coefs_row = []
    neq_constr_coefs_col = []
    neq_constr_coefs_data = []
    neq_constr_rhs = []
    index_id = 0
    constr_id = 0

    #p_ij >= x_ij
    for row in range(n):
        for col in edge_dict[row]:
            if (row < col):
                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_var[(row,col)])
                neq_constr_coefs_data.append(1)

                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_cnt + edge_var[(row,col)])
                neq_constr_coefs_data.append(-1)

                neq_constr_rhs.append(0)
                constr_id += 1

    #p_ij >= -x_ij
    for row in range(n):
        for col in edge_dict[row]:
            if row < col:
                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_var[(row,col)])
                neq_constr_coefs_data.append(-1)

                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_cnt + edge_var[(row,col)])
                neq_constr_coefs_data.append(-1)

                neq_constr_rhs.append(0)
                constr_id += 1

    #p_ij <= a_ij * u_ij * yi
    for row in range(n):
        for col in edge_dict[row]:
            if row < col:
                assert a[row, col] == 1
                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_cnt + edge_var[(row,col)])
                neq_constr_coefs_data.append(1)
                neq_constr_rhs.append(u[row,col] * (y_init[row] + delta_y[row]))
                constr_id += 1

    #p_ij <= a_ij * u_ij * yj
    for row in range(n):
        for col in edge_dict[row]:
            if row < col:
                neq_constr_coefs_row.append(constr_id)
                neq_constr_coefs_col.append(edge_cnt + edge_var[(row,col)])
                neq_constr_coefs_data.append(1)
                neq_constr_rhs.append(u[row,col] * (y_init[col] + delta_y[col]))
                constr_id += 1

    assert constr_id == int(n_neq_constrs)
    neq_constr_coefs = sparse.csr_matrix((neq_constr_coefs_data, (neq_constr_coefs_row, neq_constr_coefs_col)),
                                        shape = (int(n_neq_constrs), n_variables))
    neq_constr_rhs = np.array(neq_constr_rhs)
    return int(n_neq_constrs), neq_constr_coefs, neq_constr_rhs

def get_neq_constrs_full(n, delta_y):
    """
    input: 
        n:  scalar (# of nodes)
        n_variables: scalar (# of primal variables) : 2*n**2 + 2*n
        delta_y:  vector (gurobipy integer variables)
    output: n_neq_constrs, neq_constr_coefs, neq_constr_rhs
    """
    n_variables = 2*n**2 + 2*n
    #n_neq_constrs = n**2 + 4*n*(n-1)/2 + 2*n
    n_neq_constrs = n**2 + 4*n*(n-1)/2
    assert n_neq_constrs == int(n_neq_constrs)
    #neq_constr_coefs = np.zeros((int(n_neq_constrs), n_variables))
    #neq_constr_rhs = np.zeros(int(n_neq_constrs))
    #neq_constr_rhs = {}
    neq_constr_rhs = []

    neq_constr_coefs_row = []
    neq_constr_coefs_col = []
    neq_constr_coefs_data = []
    index_id = 0
    constr_id = 0

    #p_ij >= x_ij
    for i in range(n):
        for j in range(n):
            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n*i+j)
            neq_constr_coefs_data.append(1)

            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n**2+n*i+j)
            neq_constr_coefs_data.append(-1)

            neq_constr_rhs.append(0)
            constr_id += 1

    #x_ij <= a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n*i+j)
            neq_constr_coefs_data.append(1)
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[i] + delta_y[i]))
            constr_id += 1

    # x_ij <= a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n*i+j)
            neq_constr_coefs_data.append(1)
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n*i+j)
            neq_constr_coefs_data.append(-1)
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[i] + delta_y[i]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs_row.append(constr_id)
            neq_constr_coefs_col.append(n*i+j)
            neq_constr_coefs_data.append(-1)
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1
            
    """
    #p_ij >= x_ij
    for i in range(n):
        for j in range(n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            neq_constr_coefs[constr_id, n**2+n*i+j] = -1
            #neq_constr_rhs[constr_id] = 0
            neq_constr_rhs.append(0)
            constr_id += 1

    #x_ij <= a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            #neq_constr_rhs[constr_id] = a[i,j] * u[i,j] * (y_init[i] + delta_y[i])
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[i] + delta_y[i]))
            constr_id += 1

    # x_ij <= a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            #neq_constr_rhs[constr_id] = a[i,j] * u[i,j] * (y_init[j] + delta_y[j])
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = -1
            #neq_constr_rhs[constr_id] = a[i,j] * u[i,j] * (y_init[i] + delta_y[i])
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = -1
            #neq_constr_rhs[constr_id] = a[i,j] * u[i,j] * (y_init[j] + delta_y[j])
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1
    """
    """
    #delta_neg >= 0
    for i in range(n):
        neq_constr_coefs[constr_id, 2*n**2+2*i] = -1
        #neq_constr_rhs[constr_id] = 0
        neq_constr_rhs.append(0)
        constr_id += 1

    #delta_pos >= 0
    for i in range(n):
        neq_constr_coefs[constr_id, 2*n**2+2*i+1] = -1
        #neq_constr_rhs[constr_id] = 0
        neq_constr_rhs.append(0)
        constr_id += 1
    """
    neq_constr_coefs = sparse.csr_matrix((neq_constr_coefs_data, (neq_constr_coefs_row, neq_constr_coefs_col)),
                                        shape = (int(n_neq_constrs), n_variables))
    neq_constr_rhs = np.array(neq_constr_rhs)
    return int(n_neq_constrs), neq_constr_coefs, neq_constr_rhs

def get_dual_problem(n, delta_y=None):
    """
    max -c1 z1 + -c2 z2
    s.t. [A11, A12] z == -b1
            [A21, A22] z >= -b2
            z1 >= 0
    """
    if delta_y is None:
        delta_y = np.zeros(n)
    #n_variables_unrestricted =  2*n**2
    n_variables_unrestricted = 2 * edge_cnt
    n_variables_positive = 2*n
    _, neq_constr_coefs, neq_constr_rhs = get_neq_constrs(n, delta_y)
    _, eq_constr_coefs, eq_constr_rhs = get_eq_constrs(n)
    #A11 = (neq_constr_coefs[:-n_variables_positive,:n_variables_unrestricted]).T
    #A21 = (neq_constr_coefs[:-n_variables_positive,n_variables_unrestricted:]).T
    A11 = (neq_constr_coefs[:,:n_variables_unrestricted]).T
    A21 = (neq_constr_coefs[:,n_variables_unrestricted:]).T
    A12 = (eq_constr_coefs[:,:n_variables_unrestricted]).T
    A22 = (eq_constr_coefs[:,n_variables_unrestricted:]).T
    b1 = obj_coefs[:n_variables_unrestricted]
    b2 = obj_coefs[n_variables_unrestricted:]
    #c1 = neq_constr_rhs[:-n_variables_positive]
    c1 = neq_constr_rhs
    c2 = eq_constr_rhs

    dual_obj_coefs1 = c1
    dual_obj_coefs2 = c2
    dual_obj_coefs = np.concatenate([dual_obj_coefs1, dual_obj_coefs2])

    #dual_eq_constr_coefs = np.concatenate([A11, A12], axis = 1)
    #dual_neq_constr_coefs = np.concatenate([-A21, -A22], axis = 1)

    dual_eq_constr_coefs = scipy.sparse.hstack((A11,A12), format='csr')
    dual_neq_constr_coefs = scipy.sparse.hstack((-A21, -A22), format='csr')

    dual_eq_constr_rhs = -b1
    dual_neq_constr_rhs = b2

    #dual_pos_constr_coefs = -np.eye(A21.shape[1], dual_neq_constr_coefs.shape[1])
    dual_pos_constr_coefs = -scipy.sparse.spdiags(np.ones(A21.shape[1]), 0, A21.shape[1], dual_neq_constr_coefs.shape[1], 'csr')
    dual_pos_constr_rhs = np.zeros(A21.shape[1])
    #dual_neq_constr_coefs = np.concatenate([dual_neq_constr_coefs, dual_pos_constr_coefs], axis = 0)
    dual_neq_constr_coefs = scipy.sparse.vstack((dual_neq_constr_coefs, dual_pos_constr_coefs), format='csr')
    dual_neq_constr_rhs = np.concatenate([dual_neq_constr_rhs, dual_pos_constr_rhs], axis=0)

    dual_n_pos_constrs = A21.shape[1]
    dual_n_variables = dual_eq_constr_coefs.shape[1]
    dual_n_eq_constrs =  dual_eq_constr_coefs.shape[0]
    dual_n_neq_constrs =  dual_neq_constr_coefs.shape[0]
    return dual_n_variables, dual_obj_coefs1.shape[0], dual_obj_coefs2.shape[0],\
            dual_obj_coefs,\
            dual_eq_constr_coefs, dual_neq_constr_coefs,\
            dual_eq_constr_rhs, dual_neq_constr_rhs,\
            dual_n_eq_constrs, dual_n_neq_constrs, dual_n_pos_constrs

def phi_y(y):
    """
    input
        y: list of gurobi varaibles
    output:
        dual_obj_coefs: coefficients of z's
    """
    n = len(y)
    n_variables_positive = 2*n
    n_neq_constrs , _ , neq_constr_rhs = get_neq_constrs(n, y)
    _ , _ , eq_constr_rhs = get_eq_constrs(n)

    #c1 = np.array([neq_constr_rhs[i] for i in range(n_neq_constrs-n_variables_positive)])
    c1 = neq_constr_rhs
    c2 = eq_constr_rhs

    dual_obj_coefs1 = c1
    dual_obj_coefs2 = c2
    dual_obj_coefs = np.concatenate([dual_obj_coefs1, dual_obj_coefs2])
    return dual_obj_coefs


def topk_y(z_iter, y_init):
    """
    m=Model('mip_0')
    m.Params.LogToConsole = 0
    delta_y = {}
    for i in range(y.shape[0]):
        delta_y[i] = m.addVar(name="y_{}".format(i), vtype=GRB.BINARY)
    m.addConstr((quicksum(delta_y[i] for i in range(len(delta_y))) <= budget), name="cy_2")
    m.addConstrs((y[i] + delta_y[i] <= 1 for i in range(y.shape[0])), name="cy_1")
    dual_obj_coefs = phi_y(delta_y)
    """
    m=Model('mip_0')
    m.Params.LogToConsole = 0
    y = {}
    for i in range(y_init.shape[0]):
        y[i] = m.addVar(name="y_{}".format(i), vtype=GRB.BINARY)
    m.addConstr((quicksum(y[i] for i in range(len(y))) <= budget), name="cy_2")
    m.addConstrs((y_init[i] + y[i] <= 1 for i in range(y_init.shape[0])), name="cy_1")
    dual_obj_coefs = phi_y(y)
    assert dual_obj_coefs.shape[0] == z_iter.shape[0]
    m.setObjective(-quicksum(dual_obj_coefs[i] * z_iter[i] for i in range(dual_obj_coefs.shape[0])), GRB.MINIMIZE)
    #print("coef of z:", dual_obj_coefs)
    #print("obj_expr_elements:", [dual_obj_coefs[i] * z_iter[i] for i in range(dual_obj_coefs.shape[0])])
    m.update()
    #print("obj_expr:", m.getObjective())
    m.optimize()
    m_i0 = m.objVal
    y_opt = np.empty(y_init.shape[0])
    for v in m.getVars():
        if (v.varName[0] == 'y'):
            idx = int(v.varName.split('_')[-1])
            y_opt[idx] = v.x
        else:
            assert False
    subgrad = -phi_y(y_opt)
    return m_i0, y_opt, subgrad

def hp_proj(u, coeff, rhs, is_neq):
    coeff_u = coeff.multiply(u).sum()
    coeff_coeff = coeff.multiply(coeff).sum()
    if is_neq:
        if coeff_u <= rhs:
            return u
    k = (rhs - coeff_u) / (coeff_coeff)
    u += k * coeff.toarray()[0]
    return u

def hp_proj_multi(args):
    dual_n_constrs,x_t,gamma_t,rho,dual_constr_coefs,dual_constr_rhs, is_neq = args
    s_t = []
    for i in range(dual_n_constrs):
        s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_constr_coefs[i],
                            dual_constr_rhs[i], is_neq))
    return s_t


"""
def hp_proj_eq(args):
    dual_n_eq_constrs,x_t,gamma_t,rho,dual_eq_constr_coefs,dual_eq_constr_rhs = args[0],args[1],args[2],args[3],args[4],args[5]
    print(type(args[0]))
    print(type(dual_n_eq_constrs))
    s_t = []
    for i in range(dual_n_eq_constrs):
        s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_eq_constr_coefs[i],
                            dual_eq_constr_rhs[i], False))
    return s_t

def hp_proj_neq(args):
    dual_n_neq_constrs,x_t,gamma_t,rho,dual_neq_constr_coefs,dual_neq_constr_rhs = args[0],args[1],args[2],args[3],args[4],args[5]
    s_t = []
    for i in range(dual_n_neq_constrs):
        s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_neq_constr_coefs[i],
                            dual_neq_constr_rhs[i], is_neq = True))
    return s_t
"""

def presolve_dual():
    """
    delta_y_idx = np.random.choice(np.arange(intact_num, n), size=budget, replace=False,)
    delta_y = np.zeros(n)
    delta_y[delta_y_idx] = 1
    """
    dual_n_variables, dual_n_obj_coefs1, _, dual_obj_coefs,\
            dual_eq_constr_coefs, dual_neq_constr_coefs,\
            dual_eq_constr_rhs, dual_neq_constr_rhs,\
            dual_n_eq_constrs, dual_n_neq_constrs, _ = get_dual_problem(n, delta_y_init)

    m=Model('mip_relax')
    m.Params.LogToConsole = 0

    x = {}
    for i in range(dual_n_obj_coefs1):
        x[i] = m.addVar(lb = 0, name="z_{}".format(i))
    for i in range(dual_n_obj_coefs1, dual_n_variables):
        x[i] = m.addVar(lb = -GRB.INFINITY, name="z_{}".format(i))

    assert len(x) == dual_n_variables
    assert len(x) == dual_obj_coefs.shape[0]
    m.addConstrs((quicksum(dual_eq_constr_coefs[i,j] * x[j] for j in range(dual_n_variables))== dual_eq_constr_rhs[i])
                                                    for i in range(dual_n_eq_constrs))
    m.addConstrs((quicksum(dual_neq_constr_coefs[i,j] * x[j] for j in range(dual_n_variables))<= dual_neq_constr_rhs[i])
                                                    for i in range(dual_n_neq_constrs))

    m.setObjective(-quicksum(dual_obj_coefs[i] * x[i] for i in range(dual_obj_coefs.shape[0])), GRB.MAXIMIZE)
    m.update()
    m.optimize()
    z_iter = np.array([v.x for v in m.getVars()])
    return z_iter


delta_y_idx_init = np.random.choice(np.arange(intact_num, n), size=budget, replace=False,)
delta_y_init = np.zeros(n)
delta_y_init[delta_y_idx_init] = 1

y_init = y.copy()
"""
n_variables = n**2 + n**2 + n + n
obj_coefs = np.zeros(n_variables)
obj_coefs[n**2:2*n**2] = transmit_cost
obj_coefs[2*n**2:] = 1
"""
n_variables = 2 * edge_cnt + n + n
obj_coefs = np.zeros(n_variables)
obj_coefs[edge_cnt:2*edge_cnt] = reduced_transmit_cost
obj_coefs[2*edge_cnt:] = 1

z_init_path = "problem_set/{}/{}/z_init.npy".format(dataset, sample_dir)
if os.path.exists(z_init_path):
    print("Using cached z init.")
    z_init = np.load(z_init_path)
else:
    print("Presolve_dual start.")
    z_init = presolve_dual()
    np.save("problem_set/{}/{}/z_init.npy".format(dataset, sample_dir), z_init)
    print("Presolve_dual ends.")


def presolve_primal():
    m=Model('mip_relax')
    m.Params.LogToConsole = 0

    x = {}

    n_variables_unrestricted =  2*edge_cnt #2*n**2
    n_variables_positive = 2*n

    for i in range(n_variables_unrestricted):
        x[i] = m.addVar(lb = -GRB.INFINITY, name="x_{}".format(i))
    for i in range(n_variables_unrestricted, n_variables):
        x[i] = m.addVar(lb = 0, name="x_{}".format(i))

    delta_y = {}
    for i in range(n):
        delta_y[i] = m.addVar(name="y_{}".format(i), vtype=GRB.BINARY)
    m.addConstrs((y[i] + delta_y[i] <= 1 for i in range(n)), name ='cy_1')
    m.addConstr((quicksum([delta_y[i] for i in range(n)]) <= budget), name='cy_2')

    n_neq_constrs, neq_constr_coefs, neq_constr_rhs = get_neq_constrs(n, delta_y)
    n_eq_constrs, eq_constr_coefs, eq_constr_rhs = get_eq_constrs(n)

    m.addConstrs((quicksum(eq_constr_coefs[i,j] * x[j] for j in range(n_variables))== eq_constr_rhs[i])
                                                    for i in range(int(n_eq_constrs)))
    m.addConstrs((quicksum(neq_constr_coefs[i,j] * x[j] for j in range(n_variables))<= neq_constr_rhs[i])
                                                    for i in range(int(n_neq_constrs)))
    m.setObjective(quicksum(obj_coefs[i] * x[i] for i in range(n_variables)),GRB.MINIMIZE)
    m.update()
    m.optimize()

    y_arr = np.zeros(n)
    var_arr = np.zeros(n_variables)
    for v in m.getVars():
        if (v.varName[0] == 'y'):
            idx = int(v.varName.split('_')[-1])
            y_arr[idx] = v.x
        if (v.varName[0] == 'x'):
            idx = int(v.varName.split('_')[-1])
            var_arr[idx] = v.x
    return var_arr, y_arr, m.objVal

obj_gt_path = "problem_set/{}/{}/obj_gt.npy".format(dataset, sample_dir)
if os.path.exists(obj_gt_path):
    print("Using cached optimal value.")
    obj_gt = np.load(obj_gt_path)
else:
    gurobi_start_time = time.time()
    var_gt, y_gt, obj_gt = presolve_primal()
    np.save("problem_set/{}/{}/obj_gt.npy".format(dataset, sample_dir), obj_gt)
    #np.save("problem_set/{}/{}/var_rand_disrupt.npy".format(dataset, sample_dir), var_gt)
    #np.save("problem_set/{}/{}/y_worst_case.npy".format(dataset, sample_dir), y_gt)
    print("gurobi time: {}".format(time.time()-gurobi_start_time))

print("optimal objective:", obj_gt)


dual_n_variables, dual_n_obj_coefs1, _,_,\
        dual_eq_constr_coefs, dual_neq_constr_coefs,\
        dual_eq_constr_rhs, dual_neq_constr_rhs,\
        dual_n_eq_constrs, dual_n_neq_constrs, dual_n_pos_constrs = get_dual_problem(n)
print(dual_n_variables)
print(dual_eq_constr_coefs.shape[0])
print(dual_neq_constr_coefs.shape[0])
print(dual_neq_constr_coefs.shape[1])
print(dual_n_pos_constrs)

"""
np.save("problem_set/{}/{}/dual_eq_constr_coefs.npy".format(dataset, sample_dir), dual_eq_constr_coefs)
np.save("problem_set/{}/{}/dual_neq_constr_coefs.npy".format(dataset,sample_dir), dual_neq_constr_coefs)
np.save("problem_set/{}/{}/dual_eq_constr_rhs.npy".format(dataset,sample_dir), dual_eq_constr_rhs)
np.save("problem_set/{}/{}/dual_neq_constr_rhs.npy".format(dataset,sample_dir), dual_neq_constr_rhs)
np.save("problem_set/{}/{}/dual_n_pos_constrs.npy".format(dataset,sample_dir), dual_n_pos_constrs)
"""

def test0(rho, beta,alpha,iter_num):
    z_list = [z_init.copy()]
    objVal_list = []
    constr_list = []
    time_list = []
    x_list = [z_init.copy()]
    n_constrs = dual_n_eq_constrs + dual_n_neq_constrs
    gamma_t = [multiprocessing.sharedctypes.RawArray(ctypes.c_float, z_init.shape[0]) for _ in range(n_constrs)]
    g_t = np.zeros_like(z_init)
    grad_sum = 0
    gamma_t_sum = 0
    s_t_sum = 0
    print("Solving {} constraints, {} dim with dual averaging".format(n_constrs, z_init.shape[0]))
       
    start_time = time.time()
    for it in range(1, iter_num):
        print("===Iter:{}===".format(it))
        x_t = x_list[-1]
        if args.decay_step:
            coef_sum = 1/np.sqrt(it)
        else:
            coef_sum = 1/np.sqrt(iter_num)
        for i in range(1):
            topk_start_time = time.time()
            objVal, y_opt, subgrad = topk_y(x_t, y_init)
            print("subgradient time: {}".format(time.time() - topk_start_time))
            g_t = subgrad
            grad_sum = g_t
            x_t = x_t + (coef_sum * alpha * grad_sum + gamma_t_sum + s_t_sum) / beta
        x_list = [x_t]        
        # s update
        projection_start_time = time.time()
        """
        s_t_args = zip(
                    list(range(dual_n_eq_constrs)) + list(range(dual_n_neq_constrs)),
                    list(np.zeros(dual_n_eq_constrs, dtype=np.int32)) + list(np.ones(dual_n_neq_constrs, dtype=np.int32))
                    )
        """
        #zip_time_end = time.time()
        #print("zip time:", zip_time_end - projection_start_time)
        #print(len(list(s_t_args)))
        #global hp_proj_single
        """
        def hp_proj_single(index , is_neq):
            #index , is_neq = args
            if is_neq:
                return hp_proj(x_t - gamma_t[index + dual_n_eq_constrs]/rho, dual_neq_constr_coefs[index],
                                    dual_neq_constr_rhs[index], is_neq = True)
            else:
                return hp_proj(x_t - gamma_t[index]/rho, dual_eq_constr_coefs[index],
                                    dual_eq_constr_rhs[index], is_neq = False)
        
        pool = multiprocessing.Pool(processes=16)
        s_t = pool.map(hp_proj_single, s_t_args)
        
        s_t = list(map(hp_proj_single, 
                list(range(dual_n_eq_constrs)) + list(range(dual_n_neq_constrs)),
                list(np.zeros(dual_n_eq_constrs, dtype=np.int32)) + list(np.ones(dual_n_neq_constrs, dtype=np.int32))))
        """
        
        #s_t = np.zeros((dual_n_eq_constrs+dual_n_neq_constrs, x_t.shape[0]))
        global hp_proj_multi
        def hp_proj_multi(args):
            sub_process_start_time = time.time()
            indices , is_neqs, outer_time = args
            #s_t = []
            s_t_sum = 0
            for index, is_neq in zip(indices, is_neqs):
                if is_neq:
                    gamma_t_array = np.frombuffer(gamma_t[index + dual_n_eq_constrs], dtype=ctypes.c_float)
                    s_t_i = hp_proj(x_t - gamma_t_array/rho, dual_neq_constr_coefs[index],
                                        dual_neq_constr_rhs[index], is_neq = True)
                    s_t_x_t = rho * (s_t_i - x_t)
                    s_t_sum += s_t_x_t
                    gamma_t_array.__iadd__(s_t_x_t)
                else:
                    gamma_t_array = np.frombuffer(gamma_t[index], dtype=ctypes.c_float)
                    s_t_i = hp_proj(x_t - gamma_t_array/rho, dual_eq_constr_coefs[index],
                                        dual_eq_constr_rhs[index], is_neq = False)
                    s_t_x_t = rho * (s_t_i - x_t) 
                    s_t_sum += s_t_x_t
                    gamma_t_array.__iadd__(s_t_x_t)
                
            ret = s_t_sum
            #print(s_t[0])
            #print("sub process time: {}".format(time.time()-sub_process_start_time))
            #print("outer to now: {}".format(time.time() - outer_time))
            return ret, time.time() #np.stack(s_t, axis = 0)
        
        num_processes = 8
        arg_1 = np.concatenate((np.arange(dual_n_eq_constrs), np.arange(dual_n_neq_constrs)), axis = 0)
        arg_1 = np.array_split(arg_1, num_processes)
        arg_2 = np.concatenate((np.zeros(dual_n_eq_constrs, dtype=np.int32), np.ones(dual_n_neq_constrs, dtype=np.int32)), axis=0)
        arg_2 = np.array_split(arg_2, num_processes)
        print(dual_n_eq_constrs, dual_n_neq_constrs)
        arg_3 = np.ones(num_processes) * time.time()
        s_t_args = zip(arg_1, arg_2, arg_3)
        pool = multiprocessing.Pool(processes=num_processes)
        ret = pool.map(hp_proj_multi, s_t_args)
        #s_t = [v[0] for v in ret]
        times = [v[1] for v in ret]
        subprocess_end_time = time.time()
        print("return time: {}".format(subprocess_end_time - times[-1]))
        s_t_sum = np.sum([v[0] for v in ret], axis = 0)
        #print("concat time: {}".format(time.time() - subprocess_end_time))
        pool.close()
        
        """
        #project_x = []
        s_t_sum = 0
        for i in range(dual_n_eq_constrs):
            s_t_i = hp_proj(x_t - gamma_t[i]/rho, dual_eq_constr_coefs[i],
                                    dual_eq_constr_rhs[i], is_neq = False)
            s_t_x_t = rho * (s_t_i - x_t)
            s_t_sum += s_t_x_t
            gamma_t[i] += s_t_x_t
            #s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_eq_constr_coefs[i],
            #                        dual_eq_constr_rhs[i], is_neq = False))
            #project_x.append(hp_proj(x_t, dual_eq_constr_coefs[i],
            #                        dual_eq_constr_rhs[i], is_neq = False))
        for i in range(dual_n_neq_constrs):
            s_t_i = hp_proj(x_t - gamma_t[i+dual_n_eq_constrs]/rho, dual_neq_constr_coefs[i],
                                    dual_neq_constr_rhs[i], is_neq = True)
            s_t_x_t = rho * (s_t_i - x_t)
            s_t_sum += s_t_x_t
            gamma_t[i+dual_n_eq_constrs] += s_t_x_t
            #s_t.append(hp_proj(x_t - gamma_t[i+dual_n_eq_constrs]/rho, dual_neq_constr_coefs[i],
            #                        dual_neq_constr_rhs[i], is_neq = True))
            #project_x.append(hp_proj(x_t, dual_neq_constr_coefs[i],
            #                        dual_neq_constr_rhs[i], is_neq = True))
        """
        gamma_t_sum += s_t_sum
        #gamma_list = [gamma_t]
        print("projection time: {}".format(time.time() - projection_start_time))
        #s_list.append(s_t)
        calc_violation_start = time.time()
        print(dual_neq_constr_coefs.shape)
        print(x_t.shape)
        ag_sum_1 = np.linalg.norm((dual_neq_constr_coefs @ x_t > dual_neq_constr_rhs) * (dual_neq_constr_coefs @ x_t - dual_neq_constr_rhs)) ** 2 +\
                    np.linalg.norm(dual_eq_constr_coefs @ x_t - dual_eq_constr_rhs) ** 2
        print("ObjVal:{} CV:{}".format(-objVal, ag_sum_1))
        objVal_list.append(-objVal)
        constr_list.append(ag_sum_1)

        print("calc violation time:", time.time() - calc_violation_start)
        """
        s_list = [s_t]
        s_t_sum = 0
        # lambda update
        sum_time_start = time.time()
        for i in range(n_constrs):
            s_t_x_t = rho * (s_t[i] - x_t)
            s_t_sum += s_t_x_t
            gamma_t[i] += s_t_x_t #rho * (s_t[i] - x_t)
        gamma_t_sum += s_t_sum
        print("sum time:", time.time() - sum_time_start)
        #gamma_list.append(gamma_t)
        gamma_list = [gamma_t]
        """
        time_list.append(time.time() - start_time)
        np.save("results/{}/{}/".format(dataset, sample_dir) +\
                 "{}_rho{}_beta_{}_alpha_{}_iter_{}_obj.npy".format(file_name, rho, beta, alpha,iter_num), np.array(objVal_list))
        np.save("results/{}/{}/".format(dataset, sample_dir) +\
                 "{}_rho{}_beta_{}_alpha_{}_iter_{}_constr.npy".format(file_name, rho, beta, alpha,iter_num), np.array(constr_list))
        np.save("results/{}/{}/".format(dataset, sample_dir) +\
                 "{}_rho{}_beta_{}_alpha_{}_iter_{}_time.npy".format(file_name, rho, beta, alpha,iter_num), np.array(time_list))

    plt.figure()
    plt.plot(objVal_list)
    plt.plot(-obj_gt * np.ones_like(objVal_list))
    plt.title("mean:{:.4f} last: {:.4f} gt:{:.4f}".format(np.mean(objVal_list), objVal_list[-1], -obj_gt))
    "results/{}/{}".format(dataset, sample_dir)
    plt.savefig("results/{}/{}/".format(dataset, sample_dir) +\
                 "{}_rho{}_beta_{}_alpha_{}_iter_{}_1.png".format(file_name, rho, beta, alpha,iter_num))
    
    plt.figure()
    plt.plot(constr_list[10:])
    plt.title("min:{:.6f} last:{:.6f}".format(np.min(constr_list), constr_list[-1]))
    plt.savefig("results/{}/{}/".format(dataset, sample_dir) +\
                 "{}_rho{}_beta_{}_alpha_{}_iter_{}_2.png".format(file_name, rho, beta, alpha, iter_num))
    """"
    with open("results/{}/{}/spotlight.txt".format(dataset, sample_dir), "a") as txt_file:
        txt_file.write("rho{}_beta_{}_alpha_{}_iter_{}\n".format(rho, beta, alpha, iter_num))
        best_obj = -10
        best_vlt = 0.01
        for i in range(iter_num-1):
            if objVal_list[i] < best_obj and constr_list[i] < 0.01:
                best_obj = objVal_list[i]
                if constr_list[i] < best_vlt:
                    best_vlt = constr_list[i]
                txt_file.write("Iter: {} Objval: {} Violation: {}\n".format(i, objVal_list[i], constr_list[i]))
            elif constr_list[i] < best_vlt:
                best_vlt = constr_list[i]
                txt_file.write("Iter: {} Objval: {} Violation: {}\n".format(i, objVal_list[i], constr_list[i]))
    """
    return x_list, objVal_list, y_opt, z_list, constr_list

#_, objVal_list_1, _, _, _, _, constr_list_1 = test0()
for iter_num in [args.iter_num]:
    for rho in [args.rho]: #1
        for beta_coef in [args.beta]: #100
            beta = beta_coef * rho
            for alpha in [args.alpha]: #50
                _, objVal_list_1, _, _, constr_list_1 = test0(rho, beta,alpha, iter_num)

sys.exit()
#_, objVal_list_1, _,_, constr_list_1 = test1()
_, objVal_list_2, _,_, constr_list_2 = test1()
_, objVal_list_3, _,_, constr_list_3 = test2()

plt.figure()
plt.plot(objVal_list_1[20:])
plt.plot(objVal_list_2[20:])
plt.plot(objVal_list_3[20:])
plt.plot(-obj_gt * np.ones_like(objVal_list_1[20:]))
plt.legend(["1","2","3"])
plt.savefig(file_name + "_1.png")

plt.figure()
plt.plot(constr_list_1[20:])
plt.plot(constr_list_2[20:])
plt.plot(constr_list_3[20:])
plt.legend(["1","2","3"])
plt.savefig(file_name + "_2.png")

plt.figure()
plt.plot(objVal_list_2[20:])
plt.plot(objVal_list_3[20:])
plt.legend(["2","3"])
plt.savefig(file_name + "_3.png")

plt.figure()
plt.plot(constr_list_2[20:])
plt.plot(constr_list_3[20:])
plt.legend(["2","3"])
plt.savefig(file_name + "_4.png")
