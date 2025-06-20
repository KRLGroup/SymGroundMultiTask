import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import utils
from utils import EarlyStopping
from ReplayBuffer import ReplayBuffer
from DeepAutoma import MultiTaskProbabilisticAutoma
from envs.gridworld_multitask.Environment import GridWorldEnv_multitask, OBS_SIZE
from grounder_models import CNN_grounder, GridworldClassifier, ObjectCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parameters
num_samples = 10000
num_experiments = 1
batch_size = 32
sym_grounder_model = "ObjectCNN"

model_name = "sym_grounder"

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

buffer = ReplayBuffer()


# experiment loop (each experiment trains a different sym_grounder)
for exp in range(num_experiments):

    model_dir = os.path.join(REPO_DIR, f"storage/{model_name}_{exp}")
    os.makedirs(model_dir, exist_ok=True)

    # environment used for training (fixed)
    env = GridWorldEnv_multitask(state_type="image", max_num_steps=50, randomize_loc=False)

    # environent used for testing and logging about the symbol grounder
    test_env = GridWorldEnv_multitask(state_type="image", max_num_steps=50, randomize_loc=False)

    # create model
    sym_grounder = utils.make_grounder(sym_grounder_model, len(env.dictionary_symbols))
    sym_grounder.to(device)

    # setup optimizer (train the grounder and not the DeepDFA)
    optimizer = torch.optim.Adam(sym_grounder.parameters(), lr=0.001)
    cross_entr = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    txt_logger = utils.get_txt_logger(model_dir)

    epoch = 0
    n_won = 0
    n_failed = 0
    n_episodes = 0

    loss_values = []
    test_class_accs = []
    train_class_accs = []

    # training loop
    while n_episodes < num_samples:
        n_episodes += 1

        # reset environments
        obs, task, train_env_images, train_env_labels = env.reset()
        _, _, test_env_images, test_env_labels = test_env.reset()

        # agent starts in an empty cell (never terminates in 0 actions)
        done = False
        episode_obss = [obs]
        episode_rews = [0]

        # play the episode until termination
        while not done:
            action = env.action_space.sample()
            obs, rw, done = env.step(action)
            episode_obss.append(obs)
            episode_rews.append(rw)

        # if the rewards are all 0 there is no supervision
        if rw != 0:

            if rw == 1:
                n_won += 1
            elif rw == -1:
                n_failed += 1

            txt_logger.info(f"won {n_won} tasks and failed {n_failed} over {n_episodes}")

            # extend shorter vectors to the max length
            if len(episode_rews) < env.max_num_steps+1:
                last_rew = episode_rews[-1]
                last_obs = episode_obss[-1]
                extension_length = env.max_num_steps+1 - len(episode_rews)
                episode_rews.extend([last_rew] * extension_length)
                episode_obss.extend([last_obs] * extension_length)

            # add to the buffer
            obss = np.stack(episode_obss)
            obss = torch.tensor(obss, device=device, dtype=torch.float64)
            # obss = torch.stack(episode_obss, dim=0)
            dfa_trans = task.transitions
            dfa_rew = task.rewards
            rews = torch.LongTensor(episode_rews)
            buffer.push(obss, rews, dfa_trans, dfa_rew)

        # at each iteration train the sym_grounder after the buffer is full enough
        if len(buffer) >= 10 * batch_size:

            # sample from the buffer
            obss, rews, dfa_trans, dfa_rew = buffer.sample(batch_size)

            # build the differentiable reward machine for the task
            deepDFA = MultiTaskProbabilisticAutoma(
                batch_size = batch_size, 
                numb_of_actions = task.num_of_symbols, 
                numb_of_states = max([len(tr.keys()) for tr in dfa_trans]),
                initialization = "gaussian",
                reward_type = "ternary"
            )
            deepDFA.initFromDfas(dfa_trans, dfa_rew)

            txt_logger.info(f"\nEpoch {epoch}")
            epoch += 1

            # TRAINING

            optimizer.zero_grad()

            # obtain probability of symbols from observations with sym_grounder
            symbols = sym_grounder(obss.view(-1, 3, OBS_SIZE, OBS_SIZE))
            symbols = symbols.view(-1, env.max_num_steps+1, task.num_of_symbols)

            # predict state and reward from predicted symbols with DeepDFA
            pred_states, pred_rew = deepDFA(symbols)
            pred = pred_rew.view(-1, deepDFA.numb_of_rewards)

            # maps rewards to label
            labels = (rews + 1).view(-1)

            loss = cross_entr(pred, labels)

            # update sym_grounder
            loss.backward()
            optimizer.step()

            # LOGGING

            # collect data to compute accuracy on the train enviornment (only for logging)
            train_images = []
            train_labels = []
            for c in range(7):
                for r in range(7):
                    train_images.append(train_env_images[r, c])
                    train_labels.append(train_env_labels[r, c])
            train_images = np.stack(train_images)
            train_images = torch.tensor(train_images, device=device, dtype=torch.float64)
            # train_images = torch.stack(train_images, dim=0).to(device)
            train_labels = torch.LongTensor(train_labels).to(device)

            # collect data to compute accuracy on the test enviornment (only for logging)
            test_images = []
            test_labels = []
            for c in range(7):
                for r in range(7):
                    test_images.append(test_env_images[r, c])
                    test_labels.append(test_env_labels[r, c])
            test_images = np.stack(test_images)
            test_images = torch.tensor(test_images, device=device, dtype=torch.float64)
            # test_images = torch.stack(test_images, dim=0).to(device)
            test_labels = torch.LongTensor(test_labels).to(device)

            pred_sym_train = torch.argmax(sym_grounder(train_images), dim=-1)
            train_correct_preds = torch.sum((pred_sym_train == train_labels).long())
            train_class_acc = torch.mean((pred_sym_train == train_labels).float())

            pred_sym_test = torch.argmax(sym_grounder(test_images), dim=-1)
            test_correct_preds = torch.sum((pred_sym_test == test_labels).long())
            test_class_acc = torch.mean((pred_sym_test == test_labels).float())

            txt_logger.info(f"loss: {loss.item():.4e}")
            txt_logger.info(f"grounder TRAIN accuracy = {train_correct_preds.item()} / {pred_sym_train.shape[0]} ({train_class_acc.item():.4f})")
            txt_logger.info(f"grounder TEST accuracy = {test_correct_preds.item()} / {pred_sym_test.shape[0]} ({test_class_acc.item():.4f})")

            loss_values.append(loss.item())
            test_class_accs.append(test_class_acc.item())
            train_class_accs.append(train_class_acc.item())

            # every 10 epochs print comparison between true and predicted labels
            if epoch % 10 == 0:

                txt_logger.info("\n---")
                txt_logger.info("Comparison:")
                txt_logger.info("Train")
                txt_logger.info(f"true: {train_labels.tolist()}")
                txt_logger.info(f"pred: {pred_sym_train.tolist()}")
                txt_logger.info("Test")
                txt_logger.info(f"true: {test_labels.tolist()}")
                txt_logger.info(f"pred: {pred_sym_test.tolist()}")
                txt_logger.info("---")

            # every 100 epochs plot loss and accuracies and save the sym_grounder model
            if epoch % 100 == 0:

                plt.figure(figsize=(8, 5))
                plt.plot(loss_values, label='Loss', color='blue')
                plt.title("Training Loss Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f"loss.png"))
                plt.cla()
                plt.clf()
                plt.close()

                plt.figure(figsize=(8, 5))
                plt.plot(test_class_accs, label="Test Accuracy", color="red")
                plt.plot(train_class_accs, label="Train Accuracy", color="green")
                plt.title("Classification Accuracy Over Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.ylim(-0.1, 1.1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f"accuracy.png"))
                plt.cla()
                plt.clf()
                plt.close()

                torch.save(sym_grounder, os.path.join(model_dir, f"grounder.pth"))