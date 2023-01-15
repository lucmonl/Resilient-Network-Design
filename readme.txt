This project contains following files:

primal_dual.py: solving network resilience assessment problem using the primal-dual subgradient method
randomized_smoothing.py: solving network resilience assessment problem using the randomized-smoothed primal-dual subgradient method
network.py: solving resilience network design problem using the primal-dual subgradient method

network/
    USairport_2010/ : essential files used in dataset.ipynb
    USairport_2010_16/
        a.npy: adjacency matrix
        d.npy: demand vector
        perturb_hub_index.npy: The nodes being perturbed under hub-node perturbation
        transmit_cost.npy: unit transmission cost
        u.npy: edge capacity
    USairport_2010_34/
    USairport_2010_66/
    generate_synthetic.py: generating synthetic datasets (power law and skewed).
    dataset.ipynb: preprocess the USairport dataset

Usage Example:
python primal_dual.py --dataset USairport_2010_66 --perturb_num 12 --budget 4 --rho 1 --alpha 50 --beta 100 --iter_num 3000