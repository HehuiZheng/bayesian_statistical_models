import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from bsm.bayesian_regression import ProbabilisticGRUEnsemble
from bsm.statistical_model import BRNNStatisticalModel
from bsm.utils.general_utils import create_windowed_array
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelOutput

key = jr.PRNGKey(0)
input_dim = 12 + 6  # state 12 + action 6
output_dim = 12  # next state 12
window_size = 10
noise_level = 0.1
d_l, d_u = 0, 10

s_raw = jnp.load("../data/trunk_armV4/x.npy")
u_raw = jnp.load("../data/trunk_armV4/u.npy")

x_raw = jnp.concatenate([s_raw[:,:-1,:], u_raw], axis=-1)
y_raw = s_raw[:,1:,:]

episode = x_raw.shape[0]

x_train = []
y_train = []
for i in range(episode):
    xs = x_raw[i][4:]
    ys = y_raw[i][4:]
    x_train.append(create_windowed_array(xs, window_size=window_size))
    y_train.append(create_windowed_array(ys, window_size=window_size))

x_train = jnp.concatenate(x_train)
y_train = jnp.concatenate(y_train)
print(x_train.shape)
print(y_train.shape)
# y_train = create_windowed_array(ys, window_size=window_size)
# data_std = noise_level * jnp.ones(shape=(output_dim,))
# data = Data(inputs=x_train, outputs=y_train)