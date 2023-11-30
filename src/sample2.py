import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # グラフ出力用module

from Net import Net

BATCH_SIZE = 100
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 100
PATH = "Datasetのディレクトリpath"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

trainset = torchvision.datasets.MNIST(
    root=PATH, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=40
)  # Windows Osの方はnum_workers=1 または 0が良いかも

testset = torchvision.datasets.MNIST(
    root=PATH, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=40
)  # Windows Osの方はnum_workers=1 または 0が良いかも


device = torch.device("cuda")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()  # 損失関数の定義
optimizer = optim.SGD(
    net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
)

train_loss_value = []  # trainingのlossを保持するlist
train_acc_value = []  # trainingのaccuracyを保持するlist
test_loss_value = []  # testのlossを保持するlist
test_acc_value = []  # testのaccuracyを保持するlist

for epoch in range(EPOCH):
    print("epoch", epoch + 1)  # epoch数の出力
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    sum_loss = 0.0  # lossの合計
    sum_correct = 0  # 正解率の合計
    sum_total = 0  # dataの数の合計

    # train dataを使ってテストをする(パラメータ更新がないようになっている)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()  # lossを足していく
        _, predicted = outputs.max(1)  # 出力の最大値の添字(予想位置)を取得
        sum_total += labels.size(0)  # labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()  # 予想位置と実際の正解を比べ,正解している数だけ足す

    loss = sum_loss * BATCH_SIZE / len(trainloader.dataset)
    accuracy = float(sum_correct / sum_total)
    train_loss_value.append(loss)  # traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(accuracy)  # traindataのaccuracyをグラフ描画のためにlistに保持
    print(f"train mean loss={loss}, accuracy={accuracy}")  # lossとaccuracy出力

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    # test dataを使ってテストをする
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()

    loss = sum_loss * BATCH_SIZE / len(testloader.dataset)
    accuracy = float(sum_correct / sum_total)
    test_loss_value.append(loss)
    test_acc_value.append(accuracy)
    print(f"test  mean loss={loss}, accuracy={accuracy}")  # ログの出力

plt.figure(figsize=(6, 6))  # グラフ描画用


# 以下グラフ描画
def plot(
    train_value: list[float],
    test_value: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    file_name: str,
) -> None:
    plt.plot(range(EPOCH), train_value)
    plt.plot(range(EPOCH), test_value, c="#00ff00")
    plt.xlim(0, EPOCH)
    plt.ylim(0, 2.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(["train loss", "test loss"])
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()


plot(
    train_value=train_loss_value,
    test_value=test_loss_value,
)
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(["train loss", "test loss"])
plt.title("loss")
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(["train acc", "test acc"])
plt.title("accuracy")
plt.savefig("accuracy_image.png")
plt.clf()
