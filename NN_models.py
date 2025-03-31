import torch.nn as nn
import torch.nn.functional as F

class CNN_grounder(nn.Module):
    def __init__(self, num_symbols):
        super(CNN_grounder, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(125, 50)
        self.fc2 = nn.Linear(50, num_symbols)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 3))
        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class GridworldClassifier(nn.Module):
    def __init__(self, num_classes):  # 10 items da classificare
        super(GridworldClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Ridotto da 32 a 16 filtri
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Ridotto da 64 a 32 filtri

        self.pool = nn.MaxPool2d(2, 2)  # Pooling invariato
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Input più piccolo
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: 32x32
        x = self.pool(F.relu(self.conv2(x)))  # Output: 16x16

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x

class Linear_grounder_no_droput(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder_no_droput, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(),
            nn.Linear(hidden_size, num_output),
        )
    def forward(self, x):
         return self.grounder(x)

class Linear_grounder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.2),
            nn.Softmax(),
            nn.Linear(hidden_size, num_output),
        )
    def forward(self, x):
         return self.grounder(x)