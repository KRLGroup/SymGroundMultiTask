import torch
from DeepAutoma import MultiTaskProbabilisticAutoma
from envs.gridworld_multitask.Environment import GridWorldEnv_multitask
from NN_models import CNN_grounder, GridworldClassifier, ObjectCNN
from utils import EarlyStopping
import matplotlib.pyplot as plt
from ReplayBuffer import ReplayBuffer
import numpy as np
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# parameters
num_samples = 90000
num_experiments = 5
batch_size = 32
epoch = 0
sym_grounder_model = "ObjectCNN"

output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
os.makedirs(output_folder, exist_ok=True)

buffer = ReplayBuffer()


# experiment loop (each experiment trains a different sym_grounder)
for exp in range(num_experiments):

    # environment used for training (fixed)
    env = GridWorldEnv_multitask(state_type="image", max_num_steps=50)

    # environent used for testing and logging about the symbol grounder
    test_env = GridWorldEnv_multitask(state_type="image", max_num_steps=50)

    # collect data to compute accuracy on the train enviornment (only for logging)
    # (done here because the training environment is fixed)
    train_images = []
    train_labels = []
    for c in range(7):
        for r in range(7):
            train_images.append(env.image_locations[r,c])
            train_labels.append(env.image_labels[r,c])
    train_images = np.array(train_images)
    train_images = torch.tensor(train_images, device=device, dtype=torch.float64)
    # train_images = torch.stack(train_images, dim=0).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)

    # choose the model for the sym_grounder
    if sym_grounder_model == "CNN_grounder":
        sym_grounder = CNN_grounder(len(env.dictionary_symbols)).double().to(device)
    elif sym_grounder_model == "GridWorldClassifier":
        sym_grounder = GridworldClassifier(len(env.dictionary_symbols)).double().to(device)
    elif sym_grounder_model == "ObjectCNN":
        sym_grounder = ObjectCNN(len(env.dictionary_symbols)).double().to(device)
    else:
        raise Exception("Symbol Grounder Model '{}' NOT RECOGNIZED".format(sym_grounder_model))

    # setup optimizer (train the grounder and not the DeepDFA)
    optimizer = torch.optim.Adam(sym_grounder.parameters(), lr=0.001)
    cross_entr = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()

    n_won = 0
    n_total = 0
    loss_values =[]
    test_classification_accuracy = []
    train_classification_accuracy = []

    # training loop
    while n_total < num_samples:
        n_total += 1

        # reset environments
        obs, task, _, _ = env.reset()
        _, _, test_images_env, test_labels_env = test_env.reset()

        done = False

        # initial state has always reward 0, agent starts in an empty cell
        episode_obss = [obs]
        episode_rews = [0]

        # play the episode until termination
        while not done:
            obs, rw, done = env.step(env.action_space.sample())
            env.render()
            #print(rw)
            episode_obss.append(obs)
            episode_rews.append(rw)

        # if the episode terminates succesfully (reward 1) add it to the buffer
        if rw == 1:
            n_won += 1
            print(f"won {n_won} tasks over {n_total}")

            # extend shorter vectors to the max lenght
            if len(episode_rews) < env.max_num_steps+1:
                old_len = len(episode_rews)
                last_rew = episode_rews[-1]
                last_obs = episode_obss[-1]
                for _ in range(old_len, env.max_num_steps+1):
                    episode_rews.append(last_rew)
                    episode_obss.append(last_obs)

            # add to the buffer
            obss = np.array(episode_obss)
            obss = torch.tensor(obss, device=device, dtype=torch.float64)
            # obss = torch.stack(episode_obss, dim=0)
            dfa_trans = task.transitions
            dfa_rew = task.rewards
            rews = torch.LongTensor(episode_rews)
            buffer.push(obss, rews, dfa_trans, dfa_rew)

            # collect data to compute accuracy on the test enviornment (only for logging)
            test_images = []
            test_labels = []
            for c in range(7):
                for r in range(7):
                    test_images.append(test_images_env[r, c])
                    test_labels.append(test_labels_env[r, c])
            test_images = np.array(test_images)
            test_images = torch.tensor(test_images, device=device, dtype=torch.float64)
            # test_images = torch.stack(test_images, dim=0).to(device)
            test_labels = torch.LongTensor(test_labels).to(device)

        # at each iteration train the sym_grounder after the buffer is full enough
        if len(buffer) >= 10 * batch_size:

            # sample from the buffer
            obss, rews, dfa_trans, dfa_rew = buffer.sample(batch_size)

            # build the differentiable reward machine for the task
            mt_deepDFA = MultiTaskProbabilisticAutoma(batch_size, task.num_of_symbols, max([len(tr.keys()) for tr in dfa_trans]), 2)
            mt_deepDFA.initFromDfas(dfa_trans, dfa_rew)

            '''
            train_images = []
            train_labels = []
            for c in range(7):
                for r in range(7):
                    train_images.append(train_images_env[r, c])
                    train_labels.append(train_labels_env[r, c])
            train_images = torch.stack(train_images, dim=0).to(device)
            train_labels = torch.LongTensor(train_labels).to(device)
            '''

            print(f"\nEpoch {epoch}")
            epoch +=1

            # TRAINING

            optimizer.zero_grad()

            # predict symbols from observations with sym_grounder
            symbols = sym_grounder(obss.view(-1, 3, 64, 64))
            symbols = symbols.view(-1, env.max_num_steps+1, task.num_of_symbols)

            # predict state and reward from predicted symbols with DeepDFA
            pred_states, pred_rew = mt_deepDFA(symbols)
            pred = pred_rew.squeeze(0)

            # count occurences of reward 0 and 1
            labels = rews.view(-1)  # lista o array delle label
            class_counts = torch.bincount(labels)  # es: tensor([900, 100])
            total = class_counts.sum().item()
            print(f"class_counts: {class_counts.tolist()}")

            # compute class weights (inversely proportional)
            class_weights = total / (2.0 * class_counts.float())

            loss = cross_entr(pred.view(-1, 2), labels)
            #loss = cross_entr(pred[:,-1,:], y[:,-1])

            # update sym_grounder
            loss.backward()
            optimizer.step()

            old_loss_value = loss

            # LOGGING

            pred_sym_test = torch.argmax(sym_grounder(test_images), dim=-1)
            test_class_acc = torch.sum((pred_sym_test == test_labels).long())

            pred_sym_train = torch.argmax(sym_grounder(train_images), dim=-1)
            train_class_acc = torch.sum((pred_sym_train == train_labels).long())

            print(f"loss: {loss.item()}")
            print(f"grounder TRAIN accuracy = {train_class_acc.item()} / {pred_sym_train.shape[0]}")
            print(f"grounder TEST accuracy = {test_class_acc.item()} / {pred_sym_test.shape[0]}")

            loss_values.append(loss.item())
            test_classification_accuracy.append(test_class_acc.item())
            train_classification_accuracy.append(train_class_acc.item())

            # every 10 epochs print comparison between true and predicted labels
            if epoch % 10 == 0:
                print("\n---")
                print("Comparison:")
                print("Train")
                print(f"true: {train_labels.tolist()}")
                print(f"pred: {pred_sym_train.tolist()}")
                print("TEST TARGET/PREDICTIONS")
                print(f"true: {test_labels.tolist()}")
                print(f"pred: {pred_sym_test.tolist()}")
                print("---")

            # every 100 epochs plot loss and accuracies and save the sym_grounder model
            if epoch % 100 == 0:
                plt.plot(loss_values)
                plt.savefig(os.path.join(output_folder, f"loss_values_exp_{exp}.png"))
                plt.cla()
                plt.clf()
                plt.plot(test_classification_accuracy, color="red")
                plt.plot(train_classification_accuracy, color="green")
                plt.savefig(os.path.join(output_folder, f"class_acc_exp_{exp}.png"))
                plt.cla()
                plt.clf()

                torch.save(sym_grounder, os.path.join(output_folder, f"sym_grounder_exp_{exp}.pth"))