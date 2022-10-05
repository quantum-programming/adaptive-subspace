# shadow-tools

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
overlappedGrouping(momentum_qubit_operator,T=100).get_meas_and_p()

```
