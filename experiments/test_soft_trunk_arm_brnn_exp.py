import os
import chex
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap

from bsm.bayesian_regression.bayesian_recurrent_neural_networks.rnn_ensembles import RNNState
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.bayesian_regression.bayesian_recurrent_neural_networks.rnn_ensembles import DeterministicGRUEnsemble, \
    ProbabilisticGRUEnsemble
from bsm.utils.general_utils import create_windowed_array
import jax.random as random
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelState, StatisticalModelOutput
from bsm.statistical_model.brnn_statistical_model import BRNNStatisticalModel


def load_data(path):
    window_size = 10
    s_raw = jnp.load(os.path.join(path, "x.npy"))
    u_raw = jnp.load(os.path.join(path, "u.npy"))

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
    return x_train, y_train


def plot_3d(ys, preds_mean, save_fig=False, img_dir=None, img_name=None):
    ax = plt.figure(figsize=(32,24)).add_subplot(projection='3d')
    # ax.scatter(ys[i*100:(i+1)*100, 0], ys[i*100:(i+1)*100, 1], ys[i*100:(i+1)*100, 2], zdir='z', label='Data', color='red')
    ax.scatter(ys[:, 0], ys[:, 1],ys[:, 2], zdir='z', label='Data P1', color='red')
    ax.plot(preds_mean[:, 0], preds_mean[:, 1], preds_mean[:, 2], zdir='z', label='Mean P1', color='blue')
    ax.scatter(ys[:, 6], ys[:, 7],ys[:, 8], zdir='z', label='Data 2', color='green')
    ax.plot(preds_mean[:, 6], preds_mean[:, 7], preds_mean[:, 8], zdir='z', label='Mean P2', color='purple')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    if save_fig:
        img_path = os.path.join(img_dir, img_name)
        plt.savefig(img_path)
        plt.close()
    else:
        plt.show()
    return

def mse(W, b, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(y, y_pred):
        return jnp.inner(y-y_pred, y-y_pred) / 2.0
    # We vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

if __name__ == '__main__':

    log_dir = '/cluster/scratch/zhengh/brnn_soft_trunk_arm'
    os.makedirs(log_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    SAVE_FIG = True
    if SAVE_FIG:
        import time
        timestamp = time.time()
        result_dir = os.path.join(log_dir, str(timestamp))
        os.makedirs(result_dir, exist_ok=True)
    else:
        result_dir = None


    key = jr.PRNGKey(0)
    input_dim = 12 + 6  # state 12 + action 6
    output_dim = 12  # next state 12

    noise_level = 0.1

    window_size = 10
    s_raw = jnp.load(os.path.join('../data/trunk_armV4', "x.npy"))
    u_raw = jnp.load(os.path.join('../data/trunk_armV4', "u.npy"))

    x_raw = jnp.concatenate([s_raw[:,:-1,:], u_raw], axis=-1)
    y_raw = s_raw[:,1:,:]

    episode = x_raw.shape[0]

    split = 120

    x_train = []
    y_train = []
    for i in range(split):
        xs = x_raw[i][4:]
        ys = y_raw[i][4:]
        x_train.append(create_windowed_array(xs, window_size=window_size))
        y_train.append(create_windowed_array(ys, window_size=window_size))
    x_train = jnp.concatenate(x_train)
    y_train = jnp.concatenate(y_train)

    print("Training data loaded")

    data_std = noise_level * jnp.ones(shape=(output_dim,))  # data_std
    data = Data(inputs=x_train, outputs=y_train)


    print("Init model")

    model = BRNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=log_training,
                                beta=jnp.ones((output_dim, )),  # beta=jnp.array([1.0, 1.0]), 
                                num_particles=10, features=[64, 64, 64],
                                bnn_type=ProbabilisticGRUEnsemble, train_share=0.6, num_training_steps=1,
                                weight_decay=1e-4, hidden_state_size=20, num_cells=1)

    print("Load model state")
    statistical_model_state = None

    x_train_all = jnp.concatenate(x_raw[:split,4:], axis=0)
    y_train_all = jnp.concatenate(y_raw[:split,4:], axis=0)
    preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(x_train_all, statistical_model_state)
    # plot_3d(y_train_all, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name='train.png')

    train_mse = jnp.mean(jnp.inner(y_train_all-preds.mean, y_train_all-preds.mean) / 2.0)

    print("Train MSE: " + str(train_mse.item()))

    x_test_all = jnp.concatenate(x_raw[split:episode,4:], axis=0)
    y_test_all = jnp.concatenate(y_raw[split:episode,4:], axis=0)
    preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(x_test_all, statistical_model_state)
    # plot_3d(y_test_all, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name='test.png')

    test_mse = jnp.mean(jnp.inner(y_test_all-preds.mean, y_test_all-preds.mean) / 2.0)

    print("Test MSE: " + str(test_mse.item()))


    # Test multi-step prediction
    x_test = []
    y_test = []
    for i in range(split, episode):
        xs = x_raw[i][4:]
        ys = y_raw[i][4:]
        x_test.append(create_windowed_array(xs, window_size=window_size))
        y_test.append(create_windowed_array(ys, window_size=window_size))
    
    x_test = jnp.concatenate(x_test)
    y_test = jnp.concatenate(y_test)
    multi_step_test_mse = jnp.zeros(window_size)

    
    # def f(carry, _):
    #     preds = carry
    #     preds = vmap(model, in_axes=(0, None),
    #              out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
    #                                              statistical_model_state=None))(preds.mean, statistical_model_state)
    #     step_mse = jnp.mean(jnp.inner(y_test-preds.mean, y_test-preds.mean) / 2.0)
    #     return (preds), step_mse

    # init_carry = (x_test[:, 0, :])
    # import jax
    # (opt_state), step_mse = jax.lax.scan(f, init_carry, None, length=window_size)

    # for i in range(window_size):
    #     multi_step_test_mse[i] = jax.tree_util.tree_map(lambda x: x[i], step_mse)

    def rollout_model(model, state, test_xs, test_ys):
        pred_state = state
        x = test_xs[:, [0], :]
        for i in range(window_size):
            # preds = model(x, pred_state)

            preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(x, pred_state)

            pred_state = preds.statistical_model_state
            x = preds.mean
            multi_step_test_mse[i] = jnp.mean(jnp.inner(y_test[:, i, :]-preds.mean, y_test[:, i, :]-preds.mean) / 2.0).item()
    
    rollout_model(model, statistical_model_state, x_test, y_test)

    print("Multi-step Test MSE: ")
    print(multi_step_test_mse)

    

    for i in range(10):
        train_xs = x_raw[i,4:]
        train_ys = y_raw[i,4:]
        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(train_xs, statistical_model_state)
        plot_3d(train_ys, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name="train" + str(i).zfill(2)+'.png')

    for i in range(10):
        test_xs = x_raw[split+i,4:]
        test_ys = y_raw[split+i,4:]
        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(test_xs, statistical_model_state)
        plot_3d(test_ys, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name="test" + str(i).zfill(2)+'.png')



        