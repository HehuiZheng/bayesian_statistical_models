import os
import chex
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BayesianNeuralNet
from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import DeterministicEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import ProbabilisticEnsemble
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import DeterministicFSVGDEnsemble, ProbabilisticFSVGDEnsemble
from bsm.statistical_model.abstract_statistical_model import StatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import StatisticalModelState, StatisticalModelOutput
from bsm.statistical_model.bnn_statistical_model import BNNStatisticalModel


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

def load_episode_data(dir):
    s_raw = jnp.load(os.path.join(dir, "x.npy"))
    u_raw = jnp.load(os.path.join(dir, "u.npy"))

    x_raw = jnp.concatenate([s_raw[:,:-1,:], u_raw], axis=-1)
    y_raw = s_raw[:,1:,:]
    return x_raw, y_raw

def predict_single(model, state, xs, ys, SAVE_FIG, result_dir, img_name):
    preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(xs, state)
    plot_3d(ys, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name=img_name)
    test_mse = jnp.mean(jnp.inner(ys-preds.mean, ys-preds.mean) / 2.0)
    return test_mse.item()

def rollout_model(model, state, test_xs, test_ys, save_fig, result_dir):
    multi_step_test_mse  = []
    pred_state = state
    x = test_xs[:, 0, :]
    num_visualize = 20
    rollout_data = []
    for i in range(window_size):
        preds = vmap(model, in_axes=(0, None),
                out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                statistical_model_state=None))(x, pred_state)
        pred_state = preds.statistical_model_state
        x = jnp.concatenate([preds.mean, test_xs[:, i+1, -6:]], axis=-1) if (i+1) < window_size else None
        multi_step_test_mse.append(jnp.mean(jnp.inner(test_ys[:, i, :]-preds.mean, test_ys[:, i, :]-preds.mean) / 2.0).item())
        if save_fig:
            rollout_data.append(preds.mean[:num_visualize])

    if save_fig:
        rollout_data = jnp.stack(rollout_data, axis=1)
        for i in range(num_visualize):
            plot_3d(test_ys[i], rollout_data[i], save_fig=save_fig, img_dir=result_dir, img_name='test_multi_' + str(i).zfill(2)+'.png')
    
    return jnp.array(multi_step_test_mse)

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

if __name__ == '__main__':
    # log_dir = '/cluster/scratch/zhengh/bnn_soft_trunk_arm'
    log_dir = '/cluster/home/zhengh/OpAx_soft_robots/bayesian_statistical_models/results/soft_trunk_arm//bnn_soft_trunk_arm'
    timestamp = '1698077409.647479'
    result_dir = os.path.join(log_dir, timestamp)
    save_dir = os.path.join(result_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    SAVE_FIG = True

    key = jr.PRNGKey(0)
    input_dim = 12 + 6  # state 12 + action 6
    output_dim = 12  # next state 12
    noise_level = 0.1

    x_raw, y_raw = load_episode_data('../data/trunk_armV4')

    episode = x_raw.shape[0]
    split = 120
    data_std = noise_level * jnp.ones(shape=(output_dim,))  # data_std

    print("Creating model")
    model = BNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=False,
                                beta=jnp.ones((output_dim)),  # beta=jnp.array([1.0, 1.0]), 
                                num_particles=10, features=[64, 64, 64],
                                bnn_type=ProbabilisticFSVGDEnsemble, train_share=0.6, num_training_steps=10000,
                                weight_decay=1e-4, )

    print("Load model state")
    import pickle
    with open(os.path.join(result_dir, 'model.pkl'), "rb") as handle:
        state = pickle.load(handle)

    print("Single-step prediction")
    x_train_all = jnp.concatenate(x_raw[:split,4:], axis=0)
    y_train_all = jnp.concatenate(y_raw[:split,4:], axis=0)
    train_mse = predict_single(model, state, x_train_all, y_train_all, SAVE_FIG, save_dir, 'train.png')
    print("Train MSE: " + str(train_mse))

    x_test_all = jnp.concatenate(x_raw[split:episode,4:], axis=0)
    y_test_all = jnp.concatenate(y_raw[split:episode,4:], axis=0)
    test_mse = predict_single(model, state, x_test_all, y_test_all, SAVE_FIG, save_dir, 'test.png')
    print("Test MSE: " + str(test_mse))

    print("Visualize single-step prediction")
    for i in range(10):
        train_xs = x_raw[i,4:]
        train_ys = y_raw[i,4:]
        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(train_xs, state)
        plot_3d(train_ys, preds.mean, save_fig=SAVE_FIG, img_dir=save_dir, img_name="train_single_" + str(i).zfill(2)+'.png')

    for i in range(10):
        test_xs = x_raw[split+i,4:]
        test_ys = y_raw[split+i,4:]
        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(test_xs, state)
        plot_3d(test_ys, preds.mean, save_fig=SAVE_FIG, img_dir=save_dir, img_name="test_single_" + str(i).zfill(2)+'.png')


    print("Multi-step prediction")
    # Test multi-step prediction
    window_size = 10
    from bsm.utils.general_utils import create_windowed_array

    x_test = []
    y_test = []
    for i in range(split, episode):
        xs = x_raw[i][4:]
        ys = y_raw[i][4:]
        x_test.append(create_windowed_array(xs, window_size=window_size))
        y_test.append(create_windowed_array(ys, window_size=window_size))
    
    x_test = jnp.concatenate(x_test)
    y_test = jnp.concatenate(y_test)

    multi_step_test_mse = rollout_model(model, state, x_test, y_test, SAVE_FIG, save_dir,)

    print("Multi-step Test MSE: ")
    print(multi_step_test_mse)

    result_dict = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'multi_step_test_mse': multi_step_test_mse
    }

    import json
    with open(os.path.join(save_dir, 'mse.json'), "w") as outfile:
        json.dump(result_dict, outfile)
