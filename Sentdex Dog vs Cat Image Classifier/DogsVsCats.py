import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = False  # set to true once & set to false if no need to rebuild data

# run on gpu/cpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running device: GPU")
else:
    device = torch.device("cpu")
    print("running device: CPU")


class DogsVSCats:
    # images input = 50 * 50
    IMG_SIZE = 50

    # Locations
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    TESTING = "PetImages/Testing"

    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    cat_count = 0
    dog_count = 1

    def make_training_data(self):
        # iterates over directories (cats and dogs)
        for label in self.LABELS:
            # iterate over images, tqdm = progress bar, f = file name
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)  # path to image

                        # load images, convert to grayscale, resizes
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                        # append image data with associated class
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.cat_count += 1
                        elif label == self.DOGS:
                            self.dog_count += 1

                    # some images do not load (corrupt / empty)
                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)  # shuffles in-place training data
        np.save("training_data.npy", self.training_data)  # saves data

        print("cats: ", self.cat_count)
        print("dogs: ", self.dog_count)


# set REBUILD_DATA to true if code needs to be re-run
if REBUILD_DATA:
    dogs_v_cats = DogsVSCats()
    dogs_v_cats.make_training_data()

# re-load training data
training_data = np.load("training_data.npy", allow_pickle=True)  # allow pickle slapped in to fix bugs


# network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 2d convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)  # input = 1, output = 32 features, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # forward-pass to get data size
        # create random data to pass in to get size to pass through fc1
        x = torch.rand(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)  # target = 2 classes

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            # assigns data shape
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)  # pass through all conv layers
        x = x.view(-1, self._to_linear)  # reshapes to be flattened
        x = F.relu(self.fc1(x))  # pass through fc layer 1
        x = self.fc2(x)

        return F.softmax(x, dim=1)


net = Net().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 225.0  # img in 0-225 --> turn into 0-1

y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1  # value percent of data
val_size = int(len(X) * VAL_PCT)

train_X = X[:-val_size]  # up to -value size
train_y = y[:-val_size]

test_X = X[-val_size:]  # -value size onwards
test_y = y[-val_size:]

# modify these if needed
BATCH_SIZE = 100
EPOCHS = 10


def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCHS):
        # slices through training data
        # 0-len(train_X), take steps of batch size
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  # update


def test(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct / total, 3))


train(net)
test(net)
