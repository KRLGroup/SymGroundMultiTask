import torch
from DeepAutoma import MultiTaskProbabilisticAutoma
from Environment import GridWorldEnv_multitask
from NN_models import CNN_grounder, GridworldClassifier
from utils import EarlyStopping
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

num_samples = 9000

batch_size = 64
env = GridWorldEnv_multitask(state_type="image")
X = []
y = []

#try to classify images
images = []
for c in range(7):
    for r in range(7):
        images.append(env.image_locations[r, c])
sym_labels = torch.tensor([5, 5, 5, 2, 5, 5, 5,
                           5, 0, 4, 5, 5, 5, 5,
                           5, 5, 5, 5, 5, 0, 5,
                           3, 5, 5, 1, 5, 5, 5,
                           5, 1, 5, 5, 5, 5, 3,
                           5, 5, 5, 2, 5, 5, 5,
                           5, 5, 5, 5, 5, 4, 5
                           ]).to(device)
images = torch.stack(images, dim=0).to(device)


deepDFAs = []
sym_grounder = CNN_grounder(len(env.dictionary_symbols)).double().to(device)
#sym_grounder = GridworldClassifier(len(env.dictionary_symbols)).double().to(device)

print(torch.sum((torch.argmax(sym_grounder(images), dim=-1) == sym_labels).long()))

optimizer = torch.optim.Adam(sym_grounder.parameters(), lr=0.001)
cross_entr = torch.nn.CrossEntropyLoss()
optimizer.zero_grad()
won = 0
i_task = 0

loss_values =[]
classification_accuracy = []
task_set_trans = []
task_set_rew = []

while i_task < num_samples:
    while won < batch_size:
        i_task += 1

        obs, task = env.reset()
        #print(task.__dict__)
        #print("----------------------------")

        #do one episode
        done = False
        episode_obss = []
        episode_rews = []
        while not done:
            obs, rw, done = env.step(env.action_space.sample())
            env.render()
            #print(rw)
            episode_obss.append(obs)
            episode_rews.append(rw)
        if rw == 1:
            won+= 1
            print(f"won {won} tasks over {i_task}")

            if len(episode_rews) < env.max_num_steps:
                old_len = len(episode_rews)
                last_rew = episode_rews[-1]
                last_obs = episode_obss[-1]
                for _ in range(old_len, env.max_num_steps):
                    episode_rews.append(last_rew)
                    episode_obss.append(last_obs)

            x = torch.stack(episode_obss, dim=0).to(device)
            X.append(x)
            task_set_trans.append(task.transitions)
            task_set_rew.append(task.rewards)
            y.append(episode_rews)

    X = torch.stack(X, dim= 0)
    y = torch.LongTensor(y).to(device)

    mt_deepDFA = MultiTaskProbabilisticAutoma(batch_size, task.num_of_symbols, max([len(tr.keys()) for tr in task_set_trans]), 2)
    mt_deepDFA.initFromDfas(task_set_trans, task_set_rew)
    task_set_trans = []
    task_set_rew = []
    old_loss_value = 100
    early_stopping = EarlyStopping()
    for epoch in range(80):
        print(f"Epoch {epoch}")
        optimizer.zero_grad()
        symbols = sym_grounder(X.view(-1, 3, 64, 64))
        symbols = symbols.view(-1, env.max_num_steps, task.num_of_symbols)
        pred_states, pred_rew = mt_deepDFA(symbols)
        pred = pred_rew.squeeze(0)

        loss = cross_entr(pred.view(-1, 2), y.view(-1))
        #loss = cross_entr(pred[:,-1,:], y[:,-1])

        loss.backward()
        optimizer.step()

        pred_sym = torch.argmax(sym_grounder(images), dim=-1)
        class_acc = torch.sum((pred_sym == sym_labels).long())
        #print(pred_sym)
        #print(sym_labels)
        print(f"loss: {loss.item()}")
        print(f"correct symbols= {class_acc.item()}/ 49")

        loss_values.append(loss.item())
        classification_accuracy.append(class_acc.item())
        if early_stopping(loss):
            break
        old_loss_value = loss
    print(sym_labels)
    print(pred_sym)

    won = 0
    X = []
    y = []
    deepDFAs = []


plt.plot(loss_values)
plt.savefig("loss_values.png")
plt.cla()
plt.clf()
plt.plot(classification_accuracy)
plt.savefig("class_acc.png")

torch.save(sym_grounder, "sym_grounder.pth")

