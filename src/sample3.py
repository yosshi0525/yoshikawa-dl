import matplotlib.pyplot as plt  # グラフ出力用module

from Net import Net
from Learning import Learning

from params import EPOCH

PATH = "data/sample"

net = Net()
learning = Learning(net)
learning.learn()
learning.save()

train_loss_value = learning.train_loss_value
train_acc_value = learning.train_acc_value
test_loss_value = learning.test_loss_value
test_acc_value = learning.test_acc_value

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


# plot(
#     train_value=train_loss_value,
#     test_value=test_loss_value,
#     title="loss",
#     xlabel="EPOCH",
#     ylabel="LOSS",
#     file_name="loss_image.png",
# )
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(["train loss", "test loss"])
plt.title("loss")
plt.savefig(f"{PATH}/loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c="#00ff00")
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(["train acc", "test acc"])
plt.title("accuracy")
plt.savefig(f"{PATH}/accuracy_image.png")
plt.clf()
