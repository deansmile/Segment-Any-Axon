''' This yields only 67% accuracy, so it is not reliable at all. '''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from scipy import ndimage
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 10, 5)
        self.fc1 = nn.Linear(10 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x


def get_bounding_box(mask):
    # get bounding box from mask, get minimum and maximum x and y
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def train(num_epochs, dataset, net):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trained = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch, data in tqdm(enumerate(dataset)):
            if epoch == 0:
                if np.random.randint(5) == 0:  # sample 4/5 for training
                    trained.append(False)
                    continue
                trained.append(True)
            elif not trained[batch]:
                continue

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch}: running loss {running_loss}')

    print('Finished Training')
    return net, trained


def test(model, dataloader, trained, batch_size):
    print("Start testing")
    correct = 0
    total = 0
    batches_passed = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            batches_passed += 1
            if trained[batches_passed - 1]:
                continue
            input, label = data[0].cuda(), data[1].cuda()
            output = model(input)
            for i in range(batch_size):
                if label[i].argmax() == output[i].argmax():
                    correct += 1
            total += batch_size
    print(f'Accuracy of the network on the {total} test images: {correct}/{total} = {100 * correct / total}%')


class MaskDataset(IterableDataset):
    def __init__(self, root):
        self.root = root

    def transform(self, image):
        '''crop the image with bounding box and resize to 32x32'''
        bbox = get_bounding_box(image)
        image = Image.fromarray(image)
        image = image.crop(bbox)
        image = image.resize((32, 32))
        return np.array(image)


    def __iter__(self):
        for filename in os.listdir(self.root):
            if not filename.endswith('.tif'):
                continue
            image = tifffile.imread(os.path.join(self.root, filename))
            axons = image == 2
            labled_array, num_instances = ndimage.label(axons)
            for i in range(1, num_instances + 1):  # 0 is background
                yield np.expand_dims(self.transform(labled_array == i).astype(np.float32), 0), torch.tensor([0, 1],
                                                                                                            dtype=torch.float32)

            myelin = image == 1
            labeled_array, num_instances = ndimage.label(myelin)
            for i in range(1, num_instances + 1):
                yield np.expand_dims(self.transform(labeled_array == i).astype(np.float32), 0), torch.tensor([1, 0],
                                                                                                             dtype=torch.float32)


def main():
    data_path = Path('data', 'mask_tif').absolute()
    batch_size = 16

    dataset = MaskDataset(str(data_path))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # model = Net().load_state_dict(torch.load('mask2class.pth')).cuda()
    model = Net().cuda()
    model, trained = train(3, dataloader, model)
    torch.save(model.state_dict(), 'mask2class.pth')
    np.save('trained.npy', np.array(trained))
    test(model, dataloader, trained, batch_size)


if __name__ == '__main__':
    main()
