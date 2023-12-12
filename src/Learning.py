import datetime
import os
import numpy as np

import torch
from torch import device, nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize

from params import BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, EPOCH, PATH

num_workers = os.cpu_count() or 1

transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root=PATH, train=True, transform=transform, download=True)
train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
)

testset = datasets.MNIST(root=PATH, train=False, transform=transform, download=True)
test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
)


class Learning:
    def __init__(self, net: nn.Module) -> None:
        self.device = device("cuda")
        self.net: nn.Module = net.to(self.device)
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        self.loss_function = nn.CrossEntropyLoss()  # 損失関数の定義
        self.reset_data()

    # エポックごとに学習
    def mini_batch_learning(self):
        for inputs, labels in train_loader:
            inputs: Tensor = inputs.to(self.device)
            labels: Tensor = labels.to(self.device)
            self.optimizer.zero_grad()  # 勾配を明示的に0に初期化
            outputs: Tensor = self.net(inputs)  # 順伝播
            loss: Tensor = self.loss_function(outputs, labels)  # 誤差の計算
            loss.backward()  # 誤差逆伝播法で重みを変更
            self.optimizer.step()  # 次のステップ

    # エポックごとにテスト
    def mini_batch_test(
        self, loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> tuple[float, float]:
        sum_loss = 0  # lossの合計
        correct_count = 0  # 正解数

        for inputs, labels in loader:
            inputs: Tensor = inputs.to(self.device)
            labels: Tensor = labels.to(self.device)
            self.optimizer.zero_grad()  # 勾配を明示的に0に初期化
            outputs: Tensor = self.net(inputs)
            loss: Tensor = self.loss_function(outputs, labels)  # 誤差の計算
            sum_loss += loss.item()  # lossを足していく
            predicted: Tensor = outputs.max(1)[1]  # 出力の最大値の添字(予想位置)を取得 -> AIが予想した数字
            correct_count += (predicted == labels).sum().item()  # 予想が正解であればカウント

        ave_loss = sum_loss / BATCH_SIZE
        accuracy = correct_count / (BATCH_SIZE * len(loader))

        return ave_loss, accuracy

    def reset_data(self):
        self.train_loss_value = np.empty(EPOCH)
        self.train_acc_value = np.empty(EPOCH)
        self.test_loss_value = np.empty(EPOCH)
        self.test_acc_value = np.empty(EPOCH)

    def learn(self):
        self.reset_data()

        # 学習開始のログ出力
        starting_time = datetime.datetime.now()
        print(f"start learning at {starting_time.strftime('%H:%M:%S')}")

        # エポック数だけ学習
        for epoch in range(EPOCH):
            print("epoch", epoch + 1)  # エポック数の出力

            # 学習
            self.mini_batch_learning()

            # train dataを使ってテストをする(パラメータ更新がないようになっている)
            loss, accuracy = self.mini_batch_test(train_loader)
            self.train_loss_value[epoch] = loss
            self.train_acc_value[epoch] = accuracy
            print(f"train mean loss={loss}, accuracy={accuracy}")  # lossとaccuracy出力

            # test dataを使ってテストをする
            loss, accuracy = self.mini_batch_test(test_loader)
            self.test_loss_value[epoch] = loss
            self.test_acc_value[epoch] = accuracy
            print(f"test  mean loss={loss}, accuracy={accuracy}")  # ログの出力

        # 学習終了のログ出力
        finish_time = datetime.datetime.now()
        print(f"learning finished at {finish_time.strftime('%H:%M:%S')}")
        print(f"took {finish_time - starting_time}")

    def save(self):
        torch.save(self.net.state_dict(), f"{PATH}/model.pth")
