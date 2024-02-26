# Adaptive measurement strategy for subspace methods

## Single Iteration of quantum subspace expansion with measurement strategy
```python
params_fixed = {
    "n_trial": 10,
    "n_lev": "auto",
    "subspace": "1n",
    "spin_supspace": "up",
    "cpu_assigned": 10,
    "verbose": 0,
    "load": True,
    "write_result_matrix": True,
    "OGM_param_T": 1000,
}

method = "LBCS" # select from  ["CS", "qubit_wise_commuting","LBCS", "DCS", "OGM", "naive_LBCS"]
molecule_label, n_qubit = ("H2", 8) # select from  [("H2", 4), ("H2", 8), ("LiH", 10)]
shots = 1000
param_key = (molecule_label, n_qubit, shots, method)
print("\n", param_key)
params = {
    "molecule": molecule_label,
    "n_qubits": n_qubit,
    "shots": shots,
    "method": method,
    "suffix": timestamp(),
    **params_fixed,
}

molecule = MoleculeInfo(params)
subspace_expansion = SubspaceExpansion(params, molecule)
err, std = subspace_expansion.execute_statistics(molecule)
print({"err": err, "std": std})
```

> ('H2', 8, 1000, 'LBCS')  
> {'err': 0.1108, 'std': 0.1377}


## Adaptive quantum subspace expansion with measurement strategy
```python
energy_excited_exact, energy_excited_cisd, energies_excited = h4_experiment.simulate_qse_convergence()
```

> RUN[0-0] E excited (QSE)  -1.5081996714867638  
> :  
> RUN[9-0] E excited (QSE)  -1.5076201649425292  
> RUN[0-1] E excited (QSE)  -1.5102217623932128  
> :  
> RUN[9-1] E excited (QSE)  -1.5090517618851846  
> :  
> RUN[0-9] E excited (QSE)  -1.510351863966394  
> :  
> RUN[9-9] E excited (QSE)  -1.5098123491611533

```python
n_iter = 10

plt.rcParams["font.size"] = 15
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

plt.errorbar(
    range(0, n_iter + 1),
    np.hstack([[energy_excited_cisd], np.mean(energies_excited, axis=0)]),
    yerr=np.hstack([[0], np.std(energies_excited, ddof=1, axis=0)]),
    fmt="-o",
    color="green",
    label="QC QSE (w/ shotnoise)",
)
plt.hlines(
    y=energy_excited_cisd,
    xmin=0,
    xmax=10,
    color="Orange",
    label="CISD QSE",
    linestyles="-.",
)
plt.hlines(
    y=energy_excited_exact, xmin=0, xmax=10, color="red", label="Exact", zorder=0
)
plt.legend(loc=(0.45, 0.63), fontsize=12, edgecolor="black")
plt.ylabel("E (1st)", fontsize=18)
plt.xlabel("Number of Iterations", fontsize=18)
plt.minorticks_off()
plt.text(-3.3, -1.4856, "", fontsize=15, weight="bold")
plt.show()
```
![image](https://github.com/YumaNK/adaptive-subspace/assets/19603134/c12000f6-df76-4ead-8c02-1d480717710b)


For more detail, refer [reproduce_experiment.ipynb](https://github.com/quantum-programming/adaptive-subspace/blob/main/reproduce_experiment.ipynb).

## How to cite
If you use this code, please cite "Adaptive measurement strategy for quantum subspace methods" as follows: 
```
@article{nakamura2023adaptive,
  title={Adaptive measurement strategy for quantum subspace methods},
  author={Nakamura, Yuma and Yano, Yoshichika and Yoshioka, Nobuyuki},
  journal={arXiv preprint arXiv:2311.07893},
  year={2023}
}
```

## Licence
Copyright (c) 2024 Nobuyuki Yoshioka

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
