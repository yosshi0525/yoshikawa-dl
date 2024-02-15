import numpy as np
import pandas as pd


df = pd.read_csv("data/values.csv")


type(df["train_loss"].to_numpy())


import torch
torch.cuda.is_available()



