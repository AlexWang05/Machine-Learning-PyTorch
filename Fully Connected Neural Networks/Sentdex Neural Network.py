import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# datasets
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))

test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


# Neural Net class; inherits from nn.module
class Net(nn.Module):
    def __init__(self):
        super().__init__()  # runs initialization for nn.module

        # define layers
        # 28*28 = flattened image input; output = 64
        self.fc1 = nn.Linear(28 * 28, 64)  # fully connected net
        self.fc2 = nn.Linear(64, 64)  # takes in 64 from previous output
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # output = 10 classes/neurons (0-9)

    def forward(self, x):  # x = input/data
        # F.relu = Rectified Linear (activation function)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # softmax optimizer used for multi-class net
        return F.log_softmax(x, dim=1)  # dimension -> flat


net = Net()  # define network

optimizer = optim.Adam(net.parameters(), lr=0.001)  # learning rate = 0.01 (1e-3)

# whole passes through data
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in test_set:
        # data = batch of feature sets and labels
        X, y = data
        net.zero_grad()  # start at zero gradient
        output = net(X.view(-1, 28 * 28))  # flatten data and pass through net

        # calculate loss
        loss = F.nll_loss(output, y)
        loss.backward()  # backprop loss
        optimizer.step()  # adjust weight

correct = 0
total = 0

with torch.no_grad():
    for data in train_set:
        X, y = data
        output = net(X.view(-1, 28 * 28))

        # for every prediction, does it match target value
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("accuracy: ", round(correct / total, 3))

# for illustration
plt.imshow(X[3].view(28, 28))  # converts back to 28*28 and shows image at index
plt.show()

print(torch.argmax(net(X[3].view(-1, 784))[0]))  # prints prediction for index
