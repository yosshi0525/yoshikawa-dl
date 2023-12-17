import matplotlib.pyplot as plt
import pandas as pd

from params import PATH


def plot(csv_path, dir_path=PATH) -> None:
    df = pd.read_csv(csv_path)
    epoch = len(df)
    print(epoch)
    train_loss_value = df["train_loss"].to_numpy()
    train_acc_value = df["train_acc"].to_numpy()
    test_loss_value = df["test_loss"].to_numpy()
    test_acc_value = df["test_acc"].to_numpy()

    plt.figure(figsize=(6, 6))  # グラフ描画用

    # 以下グラフ描画
    plt.plot(train_loss_value)
    plt.plot(test_loss_value, c="#00ff00")
    plt.xlim(1, epoch)
    plt.ylim(0, 2.5)
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend(["train loss", "test loss"])
    plt.title("loss")
    plt.savefig(f"{dir_path}/loss_image.png")
    plt.clf()

    plt.plot(train_acc_value)
    plt.plot(test_acc_value, c="#00ff00")
    plt.xlim(1, epoch)
    plt.ylim(0, 1)
    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    plt.legend(["train acc", "test acc"])
    plt.title("accuracy")
    plt.savefig(f"{dir_path}/accuracy_image.png")
    plt.clf()


if __name__ == "__main__":
    plot("data/values.csv")
