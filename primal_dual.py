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
random.seed(10)
np.random.seed(0)
import scipy.sparse

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
file_name = os.path.basename(sys.argv[0])[:-3]
sys.stdout = Logger(filename = file_name + ".txt")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="USairport_2010", type=str, help="USairport_2010")
parser.add_argument("--perturb_num", default=12, type=int, help="# of perturbed nodes")
parser.add_argument("--budget", default=2, type=int, help="budget count")
parser.add_argument("--perturb_hub_node", default=False, type=bool, help="True: perturb hub nodes")
parser.add_argument("--coef_u", default=5, type=float, help="coefficient of standard capacity")
parser.add_argument("--coef_c", default=0.1, type=float, help="coefficient of transmission cost")
parser.add_argument("--dataset_id", default=0, type=int, help="dataset id 0~50")
args = parser.parse_args()
dataset =  args.dataset
print(args)

# adjacency
if dataset in ["synthetic_uniform","power_law_53","uniform_53"]:
    n = 53
    perturb_num = args.perturb_num
    intact_num = n - perturb_num
    budget = args.budget
    a = scipy.sparse.load_npz("network/{}/a_{}.npz".format(dataset, args.dataset_id)).toarray()
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
elif dataset == "USairport_2010" or dataset == "USairport_2010_76":
    perturb_num = args.perturb_num
    a = np.load("network/{}/a.npy".format(args.dataset))
    n = a.shape[0]
    intact_num = n - perturb_num
    budget = args.budget
    transmit_cost = np.load("network/{}/transmit_cost.npy".format(args.dataset)) * args.coef_c
    d = np.load("network/{}/d.npy".format(args.dataset))
    u = np.load("network/{}/u.npy".format(args.dataset)) * args.coef_u
    y = np.ones(n)
    if args.perturb_hub_node:
        perturb_index = np.load("network/{}/perturb_hub_index.npy".format(args.dataset))
    else:
        perturb_index = np.random.choice(n, perturb_num, replace=False)
    print("Perturbed Nodes: ", perturb_index)
    y[perturb_index] = 0

sample_dir = "{}_{}_{}".format(n,intact_num,budget)
os.makedirs("problem_set/{}/{}".format(dataset, sample_dir), exist_ok=True)
os.makedirs("results/{}/{}".format(dataset,"{}_".format(file_name) + sample_dir), exist_ok=True)
"""
os.makedirs("results/{}/{}".format(dataset,"{}_".format(file_name) + sample_dir), exist_ok=True)
os.makedirs("problem_set/{}/{}".format(dataset, sample_dir), exist_ok=True)

transmit_cost = np.random.random((n,n)) * cost_ub #each edge cost is calculated twice
transmit_cost = (transmit_cost + transmit_cost.T) / 2
transmit_cost = transmit_cost.flatten()


np.save("problem_set/{}/{}/d.npy".format(dataset, sample_dir), d)
np.save("problem_set/{}/{}/u.npy".format(dataset, sample_dir), u)
np.save("problem_set/{}/{}/transmit_cost.npy".format(dataset, sample_dir), transmit_cost)
np.save("problem_set/{}/{}/a.npy".format(dataset, sample_dir), a)
#np.save("problem_set/{}/{}/disrupted_nodes.npy".format(dataset, sample_dir), perturb_index)
"""


def get_eq_constrs(n):
    n_variables = 2*n**2 + 2*n
    n_eq_constrs = n*(n+1)/2 + n*(n-1)/2 + n
    assert n_eq_constrs == int(n_eq_constrs)
    eq_constr_coefs = np.zeros((int(n_eq_constrs), n_variables))
    eq_constr_rhs = np.zeros(int(n_eq_constrs))
    constr_id = 0

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
    return n_eq_constrs, eq_constr_coefs, eq_constr_rhs

def get_neq_constrs(n, delta_y):
    """
    input: 
        n:  scalar (# of nodes)
        n_variables: scalar (# of primal variables) : 2*n**2 + 2*n
        delta_y:  vector (gurobipy integer variables)
    output: n_neq_constrs, neq_constr_coefs, neq_constr_rhs
    """
    n_variables = 2*n**2 + 2*n
    n_neq_constrs = n**2 + 4*n*(n-1)/2
    assert n_neq_constrs == int(n_neq_constrs)
    neq_constr_coefs = np.zeros((int(n_neq_constrs), n_variables))
    neq_constr_rhs = []
    constr_id = 0

    #p_ij >= x_ij
    for i in range(n):
        for j in range(n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            neq_constr_coefs[constr_id, n**2+n*i+j] = -1
            neq_constr_rhs.append(0)
            constr_id += 1

    #x_ij <= a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[i] + delta_y[i]))
            constr_id += 1

    # x_ij <= a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = 1
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yi
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = -1
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[i] + delta_y[i]))
            constr_id += 1

    #x_ij >= -a_ij * u_ij * yj
    for i in range(n):
        for j in range(i+1,n):
            neq_constr_coefs[constr_id, n*i+j] = -1
            neq_constr_rhs.append(a[i,j] * u[i,j] * (y_init[j] + delta_y[j]))
            constr_id += 1

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
    n_variables_unrestricted =  2*n**2
    n_variables_positive = 2*n
    _, neq_constr_coefs, neq_constr_rhs = get_neq_constrs(n, delta_y)
    _, eq_constr_coefs, eq_constr_rhs = get_eq_constrs(n)
    A11 = (neq_constr_coefs[:,:n_variables_unrestricted]).T
    A21 = (neq_constr_coefs[:,n_variables_unrestricted:]).T
    A12 = (eq_constr_coefs[:,:n_variables_unrestricted]).T
    A22 = (eq_constr_coefs[:,n_variables_unrestricted:]).T
    b1 = obj_coefs[:n_variables_unrestricted]
    b2 = obj_coefs[n_variables_unrestricted:]
    c1 = neq_constr_rhs
    c2 = eq_constr_rhs

    dual_obj_coefs1 = c1
    dual_obj_coefs2 = c2
    dual_obj_coefs = np.concatenate([dual_obj_coefs1, dual_obj_coefs2])

    dual_eq_constr_coefs = np.concatenate([A11, A12], axis = 1)
    dual_neq_constr_coefs = np.concatenate([-A21, -A22], axis = 1)
    dual_eq_constr_rhs = -b1
    dual_neq_constr_rhs = b2

    dual_pos_constr_coefs = -np.eye(A21.shape[1], dual_neq_constr_coefs.shape[1])
    dual_pos_constr_rhs = np.zeros(A21.shape[1])
    dual_neq_constr_coefs = np.concatenate([dual_neq_constr_coefs, dual_pos_constr_coefs], axis = 0)
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

    c1 = neq_constr_rhs
    c2 = eq_constr_rhs

    dual_obj_coefs1 = c1
    dual_obj_coefs2 = c2
    dual_obj_coefs = np.concatenate([dual_obj_coefs1, dual_obj_coefs2])
    return dual_obj_coefs

def topk_y(z_iter, y_init):
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
    print("coef of z:", dual_obj_coefs)
    m.update()
    print("obj_expr:", m.getObjective())
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
    if is_neq:
        if coeff @ u <= rhs:
            return u
    k = (rhs - coeff @ u) / (coeff @ coeff)
    u += k * coeff
    return u

def hp_proj_multi(args):
    dual_n_constrs,x_t,gamma_t,rho,dual_constr_coefs,dual_constr_rhs, is_neq = args
    s_t = []
    for i in range(dual_n_constrs):
        s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_constr_coefs[i],
                            dual_constr_rhs[i], is_neq))
    return s_t


def presolve_dual():
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

"""
delta_y_idx_init = np.random.choice(np.arange(intact_num, n), size=budget, replace=False,)
delta_y_init = np.zeros(n)
delta_y_init[delta_y_idx_init] = 1
"""
y_init = y.copy()
n_variables = n**2 + n**2 + n + n
obj_coefs = np.zeros(n_variables)
obj_coefs[n**2:2*n**2] = transmit_cost
obj_coefs[2*n**2:] = 1
#print("Presolve_dual start.")
z_init = presolve_dual()
np.save("problem_set/{}/z_init.npy".format(sample_dir), z_init)
#z_init = np.load("problem_set/{}/z_init.npy".format(sample_dir))
#print("Presolve_dual ends.")

def presolve_primal():
    m=Model('mip_relax')
    m.Params.LogToConsole = 0

    x = {}

    n_variables_unrestricted =  2*n**2
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

gurobi_start_time = time.time()
var_gt, y_gt, obj_gt = presolve_primal()
print("optimal objective:", obj_gt)
print("gurobi time: {}".format(time.time()-gurobi_start_time))
#np.save("problem_set/{}/{}/var_rand_disrupt.npy".format(dataset, sample_dir), var_gt)
#np.save("problem_set/{}/{}/y_worst_case.npy".format(dataset, sample_dir), y_gt)


dual_n_variables, dual_n_obj_coefs1, _,_,\
        dual_eq_constr_coefs, dual_neq_constr_coefs,\
        dual_eq_constr_rhs, dual_neq_constr_rhs,\
        dual_n_eq_constrs, dual_n_neq_constrs, dual_n_pos_constrs = get_dual_problem(n)
print(dual_eq_constr_coefs.shape[0] + dual_neq_constr_coefs.shape[0])
np.save("problem_set/{}/dual_eq_constr_coefs.npy".format(sample_dir), dual_eq_constr_coefs)
np.save("problem_set/{}/dual_neq_constr_coefs.npy".format(sample_dir), dual_neq_constr_coefs)
np.save("problem_set/{}/dual_eq_constr_rhs.npy".format(sample_dir), dual_eq_constr_rhs)
np.save("problem_set/{}/dual_neq_constr_rhs.npy".format(sample_dir), dual_neq_constr_rhs)
np.save("problem_set/{}/dual_n_pos_constrs.npy".format(sample_dir), dual_n_pos_constrs)

def test0(rho, beta,alpha,iter_num):
    mu0 = 1
    z_list = [z_init.copy()]
    objVal_list = []
    constr_list = []
    y_list = []
    x_list = [z_init.copy()]
    theta_list = [1e8,1]
    n_constrs = dual_n_eq_constrs + dual_n_neq_constrs
    gamma_list = [np.zeros((n_constrs, z_init.shape[0]))]
    s_list = [np.outer(np.ones(n_constrs),z_init)]
    theta = 1
    g_t = np.zeros_like(z_init)
    grad_sum = 0
    gamma_t_sum = 0
    s_t_sum = 0
    coef_z_sum = 0
    print("Solving {} constraints, {} dim".format(n_constrs, z_init.shape[0]))
    
    coef_sum = 0
    for it in range(1, iter_num):
        coef_sum += 1/it
    
    for it in range(1, iter_num):
        print("===Iter:{}===".format(it))
        theta = 1
        theta_list.append(theta)
        x_t = x_list[-1]
        gamma_t = gamma_list[-1].copy()
        s_t =s_list[-1].copy()
        y_list = [(1-theta)*x_list[-1] + theta*z_list[-1]]
        x_t = y_list[-1]
        coef_sum = 1/np.sqrt(iter_num)
        for i in range(1):
            topk_start_time = time.time()
            objVal, y_opt, subgrad = topk_y(x_t, y_init)
            print("subgradient time: {}".format(time.time() - topk_start_time))
            g_t = subgrad
            grad_sum = g_t

            time1_start = time.time()
            gamma_t_sum = np.sum(gamma_t, axis=0)
            s_t_sum  = rho * np.sum(s_t - x_t, axis=0)
            print("numpy time: {}".format(time.time() - time1_start))
            x_t = x_t + (coef_sum * alpha * grad_sum + gamma_t_sum + s_t_sum) / beta
            print(x_t)
        z_t = x_t.copy()
        z_list = [z_t]
        x_list = [x_t]
        
        print("y_opt")
        print(y_opt)        
        
        # s update
        s_t = []
        projection_start_time = time.time()
        
        for i in range(dual_n_eq_constrs):
            s_t.append(hp_proj(x_t - gamma_t[i]/rho, dual_eq_constr_coefs[i],
                                    dual_eq_constr_rhs[i], is_neq = False))
        for i in range(dual_n_neq_constrs):
            s_t.append(hp_proj(x_t - gamma_t[i+dual_n_eq_constrs]/rho, dual_neq_constr_coefs[i],
                                    dual_neq_constr_rhs[i], is_neq = True))
        
        print("projection time: {}".format(time.time() - projection_start_time))
        ag_sum_0 = 0
        ag_sum_1 = 0
        for j in range(len(s_t)):
            ag_sum_0 += gamma_t[j] @ (s_t[j] - x_t)
            ag_sum_1 += (s_t[j] - x_t)@ (s_t[j] - x_t)
        print("ObjVal:{} AG:{} {} f:{}".format(-objVal, ag_sum_0, ag_sum_1, -objVal+ag_sum_0+ag_sum_1))
        objVal_list.append(-objVal)
        constr_list.append(ag_sum_1)

        s_list = [s_t]
        # lambda update
        for i in range(n_constrs):
            gamma_t[i] += rho * (s_t[i] - x_t)
        gamma_list = [gamma_t]

    plt.figure()
    plt.plot(objVal_list)
    plt.plot(-obj_gt * np.ones_like(objVal_list))
    plt.title("mean:{:.4f} last: {:.4f} gt:{:.4f}".format(np.mean(objVal_list), objVal_list[-1], -obj_gt))
    plt.savefig("results/" + "{}_".format(file_name) + sample_dir + "/" + file_name +\
                 "_rho{}_beta_{}_alpha_{}_iter_{}_1.png".format(rho, beta, alpha,iter_num))
    
    plt.figure()
    plt.plot(constr_list[10:])
    plt.title("min:{:.6f} last:{:.6f}".format(np.min(constr_list), constr_list[-1]))
    plt.savefig("results/" + "{}_".format(file_name) + sample_dir + "/" + file_name +\
                 "_rho{}_beta_{}_alpha_{}_iter_{}_2.png".format(rho, beta, alpha, iter_num))
    
    with open("results/{}_{}/spotlight.txt".format(file_name, sample_dir), "a") as txt_file:
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

    return x_list, objVal_list, y_opt, gamma_list, s_list, z_list, constr_list


for iter_num in [4000]:
    for rho in [5]:
        for beta_coef in [20]:
            beta = beta_coef * rho
            for alpha in [25]:
                _, objVal_list_1, _, _, _, _, constr_list_1 = test0(rho, beta,alpha, iter_num)
