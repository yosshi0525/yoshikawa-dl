from Net import Net
from Learning import Learning
from plot import plot


PATH = "data/sample2"

# 学習
net = Net()
learning = Learning(net)
learning.learn()
learning.save()

# プロット
plot("data/values.csv", PATH)
