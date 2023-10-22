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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    SAVE_FIG = True
    if SAVE_FIG:
        import time
        timestamp = time.time()
        result_dir = os.path.join('../results/soft_trunk_arm', str(timestamp))
        os.mkdir(result_dir)
    else:
        result_dir = None

    log_training = False
    
    import wandb
    if log_training:
        wandb.init(
            project='soft_arm_brnn',
            group='test group',
        )


    key = jr.PRNGKey(0)
    input_dim = 1
    output_dim = 2
    input_dim = 12 + 6  # state 12 + action 6
    output_dim = 12  # next state 12

    noise_level = 0.1

    # x_train, y_train = load_data('../data/trunk_arm/train')

    window_size = 50
    s_raw = jnp.load(os.path.join('../data/trunk_arm/train', "x.npy"))
    u_raw = jnp.load(os.path.join('../data/trunk_arm/train', "u.npy"))

    x_raw = jnp.concatenate([s_raw[:,:-1,:], u_raw], axis=-1)
    y_raw = s_raw[:,1:,:]

    episode = x_raw.shape[0]


    x_train = []
    y_train = []
    for i in range(100):
        xs = x_raw[i][4:]
        ys = y_raw[i][4:]
        x_train.append(create_windowed_array(xs, window_size=window_size))
        y_train.append(create_windowed_array(ys, window_size=window_size))
    x_train = jnp.concatenate(x_train)
    y_train = jnp.concatenate(y_train)

    print("Training data loaded")


    ys_train = jnp.concatenate(y_raw[:100], axis=0)

    s_raw_test = jnp.load(os.path.join('../data/trunk_arm/test', "x.npy"))
    u_raw_test = jnp.load(os.path.join('../data/trunk_arm/test', "u.npy"))

    x_raw_test = jnp.concatenate([s_raw_test[:,:-1,:], u_raw_test], axis=-1)
    y_raw_test = s_raw_test[:,1:,:]
    ys_test = jnp.concatenate(y_raw[100:], axis=0)

    plot_3d(ys_train, ys_test, save_fig=SAVE_FIG, img_dir=result_dir, img_name='dataset.png')


    data_std = noise_level * jnp.ones(shape=(output_dim,))  # data_std
    data = Data(inputs=x_train, outputs=y_train)


    print("Creating model")

    model = BRNNStatisticalModel(input_dim=input_dim, output_dim=output_dim, output_stds=data_std, logging_wandb=log_training,
                                beta=jnp.ones((output_dim, )),  # beta=jnp.array([1.0, 1.0]), 
                                num_particles=10, features=[64, 64, 64],
                                bnn_type=ProbabilisticGRUEnsemble, train_share=0.6, num_training_steps=10000,
                                weight_decay=1e-4, hidden_state_size=20, num_cells=1)

    print("Start training")
    init_model_state = model.init(key=jr.PRNGKey(0))
    statistical_model_state = model.update(model_state=init_model_state, data=data)

    print("Finish training")

    # Test on new data
    # x_test, y_test = load_data('../data/trunk_arm/test')

    s_raw_test = jnp.load(os.path.join('../data/trunk_arm/test', "x.npy"))
    u_raw_test = jnp.load(os.path.join('../data/trunk_arm/test', "u.npy"))

    x_raw_test = jnp.concatenate([s_raw_test[:,:-1,:], u_raw_test], axis=-1)
    y_raw_test = s_raw_test[:,1:,:]


 
    # for i in range(x_raw_test.shape[0]):
    #     test_xs = x_raw_test[i][4:]
    #     test_ys = y_raw_test[i][4:]

    #     preds = vmap(model, in_axes=(0, None),
    #              out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
    #                                              statistical_model_state=None))(test_xs, statistical_model_state)
        
    #     ax = plt.figure(figsize=(32,24)).add_subplot(projection='3d')
    #     # ax.scatter(test_ys[i*100:(i+1)*100, 0], test_ys[i*100:(i+1)*100, 1], test_ys[i*100:(i+1)*100, 2], zdir='z', label='Data', color='red')
    #     ax.scatter(test_ys[:, 0], test_ys[:, 1],test_ys[:, 2], zdir='z', label='Data P1', color='red')
    #     ax.plot(preds.mean[:, 0], preds.mean[:, 1], preds.mean[:, 2], zdir='z', label='Mean P1', color='blue')
    #     ax.scatter(test_ys[:, 6], test_ys[:, 7],test_ys[:, 8], zdir='z', label='Data 2', color='green')
    #     ax.plot(preds.mean[:, 6], preds.mean[:, 7], preds.mean[:, 8], zdir='z', label='Mean P2', color='purple')
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())
    #     # plt.show()
    #     img_name = os.path.join(result_dir, str(i).zfill(5)+'.png')
    #     if SAVE_FIG:
    #         plt.savefig(img_name)
    #         plt.close()
    #     else:
    #         plt.show()


    for i in range(x_train.shape[0]):
        train_xs = x_train[i]
        train_ys = y_train[i]

        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(train_xs, statistical_model_state)
        plot_3d(train_ys, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name="train" + str(i).zfill(5)+'.png')


    x_test = []
    y_test = []
    for i in range(100, episode):
        xs = x_raw[i][4:]
        ys = y_raw[i][4:]
        x_test.append(create_windowed_array(xs, window_size=window_size))
        y_test.append(create_windowed_array(ys, window_size=window_size))
    x_test = jnp.concatenate(x_test)
    y_test = jnp.concatenate(y_test)
    
    for i in range(x_test.shape[0]):
        test_xs = x_test[i]
        test_ys = y_test[i]

        preds = vmap(model, in_axes=(0, None),
                 out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
                                                 statistical_model_state=None))(test_xs, statistical_model_state)
        plot_3d(train_ys, preds.mean, save_fig=SAVE_FIG, img_dir=result_dir, img_name="test" + str(i).zfill(5)+'.png')

    



    # test_xs = jnp.concatenate(x_raw_test, axis=0)
    # # print(xs.shape)
    # print(test_xs.shape)
    # test_ys = jnp.concatenate(y_raw_test, axis=0)

    # preds = vmap(model, in_axes=(0, None),
    #              out_axes=StatisticalModelOutput(mean=0, epistemic_std=0, aleatoric_std=0,
    #                                              statistical_model_state=None))(test_xs, statistical_model_state)
    # print(preds.mean.shape)

    # for j in range(output_dim):
    #     plt.scatter(xs[:, j], ys[:, j], label='Data', color='red')
    #     plt.plot(test_xs, preds.mean[:, j], label='Mean', color='blue')
    #     plt.fill_between(test_xs[:, j],
    #                      (preds.mean[:, j] - preds.statistical_model_state.beta[j] * preds.epistemic_std[:,
    #                                                                                  j]).reshape(-1),
    #                      (preds.mean[:, j] + preds.statistical_model_state.beta[j] * preds.epistemic_std[:, j]).reshape(
    #                          -1),
    #                      label=r'$2\sigma$', alpha=0.3, color='blue')
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     plt.plot(test_xs[:, j], test_ys[:, j], label='True', color='green')
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys())
    #     plt.show()

    
    # plt.scatter(xs[:, 0], xs[:, 1], label='Data', color='red')
    # plt.plot(xs[:, 0], xs[:, 1], label='Mean', color='blue')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # plt.show()
    


        