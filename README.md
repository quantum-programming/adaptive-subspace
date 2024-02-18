# Adaptive measurement strategy for subspace methods

## OGM
```python
# prepare openfermion operator
from openfermion.hamiltonians import jellium_model
from openfermion.utils import fourier_transform, Grid
from openfermion.transforms import jordan_wigner

grid = Grid(dimensions=1, length=3, scale=1.0)
momentum_hamiltonian = jellium_model(grid, spinless=True)
momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
position_hamiltonian = fourier_transform(momentum_hamiltonian, grid, spinless=True)
position_qubit_operator = jordan_wigner(position_hamiltonian)
position_qubit_operator
```

>>19.500476387540857 [] +
-6.42058132430101 [Z0] +
-3.289868133696451 [Y0 Y1] +
-3.289868133696451 [X0 X1] +
-3.2898681336964564 [Y0 Z1 Y2] +
-3.2898681336964564 [X0 Z1 X2] +
-6.4205813243010095 [Z1] +
-3.289868133696451 [Y1 Y2] +
-3.289868133696451 [X1 X2] +
-6.42058132430101 [Z2] +
-0.07957747154594766 [Z0 Z1] +
-0.07957747154594767 [Z0 Z2] +
-0.07957747154594766 [Z1 Z2]


```python
# obtain efficient measurement set and its probability distribution according to overlapped grouping
from OverlappedGrouping.overlapped_grouping import OverlappedGrouping
overlappedGrouping(position_qubit_operator,T=100).get_meas_and_p()

```

## Classical Shadow with LBCS and OGM
```python
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion import linalg, QubitOperator

ham_h2_jw = QubitOperator()
ham_h2_jw.terms = {(): -0.8105479805373261,
 ((0, 'X'), (1, 'X'), (2, 'X'), (3, 'X')): 0.04523279994605781,
 ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')): 0.04523279994605781,
 ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')): 0.04523279994605781,
 ((0, 'Y'), (1, 'Y'), (2, 'Y'), (3, 'Y')): 0.04523279994605781,
 ((0, 'Z'),): 0.17218393261915566,
 ((0, 'Z'), (1, 'Z')): 0.1209126326177663,
 ((0, 'Z'), (2, 'Z')): 0.16892753870087912,
 ((0, 'Z'), (3, 'Z')): 0.16614543256382408,
 ((1, 'Z'),): -0.2257534922240248,
 ((1, 'Z'), (2, 'Z')): 0.16614543256382408,
 ((1, 'Z'), (3, 'Z')): 0.17464343068300447,
 ((2, 'Z'),): 0.1721839326191557,
 ((2, 'Z'), (3, 'Z')): 0.1209126326177663,
 ((3, 'Z'),): -0.2257534922240248}


num_qubits = 4
beta_eff = local_dists_optimal(ham_h2_jw, num_qubits, "diagonal", "lagrange")
# beta_eff = np.array([[0.31122347, 0.31122347, 0.37755306],
#        [0.30214128, 0.30214128, 0.39571744],
#        [0.31122347, 0.31122347, 0.37755306],
#        [0.30214128, 0.30214128, 0.39571744]])

meas_dist = QubitOperator()
meas_dist.terms = {((0, 'X'), (1, 'X'), (2, 'X'), (3, 'X')): 0.06107293172976727,
 ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')): 0.06105199949206699,
 ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')): 0.06105199949206626,
 ((0, 'Y'), (1, 'Y'), (2, 'Y'), (3, 'Y')): 0.061052009368477114,
 ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z')): 0.7557710599176223}


Nshadow_tot = 1000 # 合計のランダム測定回数
nshot_per_axis = 1
num_qubit = 4

energy, psi = get_ground_state(get_sparse_operator(ham_h2_jw))
    
Sampler = LocalPauliShadowSampler_core(num_qubit, psi, Nshadow_tot, nshot_per_axis)
print(estimate_exp(ham_h2_jw, Sampler))
print(estimate_exp_lbcs(ham_h2_jw, Sampler,beta = beta_eff))
print(estimate_exp_ogm(ham_h2_jw, Sampler,meas_dist = meas_dist))
```
> -1.8776611085760964   
-1.8798839567797456  
 -1.8496958629294746


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
