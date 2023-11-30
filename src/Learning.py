import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torch.utils.data.dataloader as dataloader
from Net import Net
from params import BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, EPOCH, PATH

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

trainset = torchvision.datasets.MNIST(
    root=PATH, train=True, transform=transform, download=True
)
trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=40,
)

testset = torchvision.datasets.MNIST(
    root=PATH, train=False, transform=transform, download=True
)
testloader = torch.utils.data.DataLoader(
    dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=40
)


device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()  # 損失関数の定義


class Learning:
    def __init__(self) -> None:
        self.net = Net().to(device)
        optimizer = optim.SGD(
            self.net.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

        self.train_loss_value = []
        self.train_acc_value = []
        self.test_loss_value = []
        self.test_acc_value = []

    # エポックごとに学習
    def epoch_learn(self):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    # エポックごとにテスト
    def epoch_test(self, loader: torch.Dataloader) -> tuple[float, float]:
        sum_loss = 0.0  # lossの合計
        sum_correct = 0  # 正解率の合計
        sum_total = 0  # dataの数の合計

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()  # 勾配を明示的に0に初期化
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)  # 誤差の計算
            sum_loss += loss.item()  # lossを足していく
            _, predicted = outputs.max(1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels.size(0)  # labelの数を足していくことでデータの総和を取る
            # 予想位置と実際の正解を比べ,正解している数だけ足す
            sum_correct += (predicted == labels).sum().item()

        loss = sum_loss * BATCH_SIZE / len(testloader.dataset)
        accuracy = float(sum_correct / sum_total)

        return loss, accuracy

    def learn(self):
        print(f"start learning at {0}")

        for epoch in range(EPOCH):
            print("epoch", epoch + 1)  # epoch数の出力

            # 学習
            self.epoch_learn()

            # train dataを使ってテストをする(パラメータ更新がないようになっている)
            loss, accuracy = self.epoch_test(trainloader)
            self.train_loss_value.append(loss)  # traindataのlossをグラフ描画のためにlistに保持
            self.train_acc_value.append(accuracy)  # traindataのaccuracyをグラフ描画のためにlistに保持
            print(f"train mean loss={loss}, accuracy={accuracy}")  # lossとaccuracy出力

            # test dataを使ってテストをする
            loss, accuracy = self.epoch_test(testloader)
            self.test_loss_value.append(loss)
            self.test_acc_value.append(accuracy)
            print(f"test  mean loss={loss}, accuracy={accuracy}")  # ログの出力

        print(f"learning finished at {0}")
