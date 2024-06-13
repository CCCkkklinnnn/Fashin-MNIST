# 导入包
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

# 基本配置
batch_size = 256
lr = 0.001
num_workers = 0
epochs = 10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 数据读入和加载
class MyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape(28, 28, 1)
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image / 255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


my_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(28),
     transforms.ToTensor()
     ]
)
df_train = pd.read_csv("./FashionMNIST/archive/fashion-mnist_train.csv")
df_test = pd.read_csv("./FashionMNIST/archive/fashion-mnist_test.csv")
train_data = MyDataset(df_train, my_transform)
test_data = MyDataset(df_test, my_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print(df_train)
print(train_data.images[0].shape)


# 构建网络模型 初始化+前向传播
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


model = net()
model.cuda()

# 损失函数
lossfuc = nn.CrossEntropyLoss()
# 优化器
optimizor = optim.Adam(model.parameters(), lr=lr)


# 训练
def train(epoch):
    model.train()
    train_loss = 0
    for image, label in train_loader:  # image[256,1,28,28] label[256]
        image, label = image.cuda(), label.cuda()
        output = model(image)
        # 梯度清零
        optimizor.zero_grad()
        # 计算loss
        loss = lossfuc(output, label)
        # 反向传播
        loss.backward()
        # 优化器迭代
        optimizor.step()
        # 计算整个loss
        train_loss += loss.item() * image.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch:{}\ttrainloss:{:.6f}'.format(epoch, train_loss))


# 评估
def evaluate(epoch):
    model.eval()
    gt = []
    pred = []
    test_loss = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.cuda(), label.cuda()
            output = model(image)
            pred_label = torch.argmax(output, 1)  # output[256,10]
            loss = lossfuc(output, label)
            test_loss += loss.item() * image.size(0)
            gt.append(label.cpu().numpy())
            pred.append(pred_label.cpu().numpy())
    gt, pred = np.concatenate(gt), np.concatenate(pred)
    acc = np.sum(gt == pred) / len(gt)
    test_loss = test_loss / len(test_loader.dataset)
    print('Epoch:{}\tacc:{:.6f}\ttestloss:{:.6f}'.format(epoch, acc, test_loss))


for epoch in range(1, epochs + 1):
    train(epoch)
    evaluate(epoch)
