
# Try it!

```
python3 drishtant_main.py
```

# Example

# This is just and example demo

```python
import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3), # First dense layer
    Tanh(), #applying activation function
    Dense(3, 1), #second dense layer and output layer
    Tanh()
]

train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.15)
```