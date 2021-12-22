# 写真に松田好花が写っているか判別するネットワークの作成
# 写っていたら1
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from augmentation import Mydataset


class KonokaNet(nn.Module):
    def __init__(self):
        super(KonokaNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))

        self.conv1 = nn.Conv2d(1, 4, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(4, 16, (3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(128*20*15, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return f.log_softmax(x, dim=1)


def pic_transform(dir_name):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    picture_set = datasets.ImageFolder(dir_name, transform)
    return picture_set


if __name__ == '__main__':
    net = KonokaNet().to('cuda')

    # 写真データの取得
    batch_size = 8
    pic_set = pic_transform('picture_sample/dataset')
    num_train = int(0.8 * len(pic_set))
    num_val = len(pic_set) - num_train
    torch.manual_seed(0)
    train, val = random_split(pic_set, [num_train, num_val])

    train = Mydataset(train)

    train_loader = DataLoader(train, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size)

    epoch_num = 20
    optimizer = optim.SGD(params=net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(epoch_num):

        # train
        net.train()
        train_loss = 0
        train_acc = 0
        train_num = 0

        for i, (images, targets) in enumerate(train_loader):

            train_num += len(images)

            images, targets = images.to('cuda'), targets.to('cuda')

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            train_acc += torch.eq(outputs.max(1)[1], targets).sum().item()
            # if i == 0:
            #     print(targets)
            #     print(outputs.max(1)[1])
            loss.backward()
            optimizer.step()

        print(train_num)
        train_loss_avg = train_loss / train_num
        train_acc_avg = train_acc / train_num
        train_loss_list.append(train_loss_avg)
        train_acc_list.append(train_acc_avg)

        # val
        net.eval()
        val_loss = 0
        val_acc = 0
        val_num = 0

        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                val_num += len(images)
                images, targets = images.to('cuda'), targets.to('cuda')
                outputs = net(images)
                # if i==0:
                #     print(outputs.max(1)[1])
                #     print(targets)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_acc += torch.eq(outputs.max(1)[1], targets).sum().item()

        val_loss = val_loss / val_num
        val_acc = val_acc / val_num
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    plt.figure()
    plt.plot(range(1, epoch_num+1), train_acc_list, label='train_acc')
    plt.plot(range(1, epoch_num+1), val_acc_list, label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # ネットワークの保存
    PATH = './Konoka_net.pt'
    torch.save(net.state_dict(), PATH)
