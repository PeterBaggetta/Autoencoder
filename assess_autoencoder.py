# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
#%%
# file = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "MyFile.csv")

# df = pd.read_csv(file)

# plt.plot(df)
# plt.show()
# plt.close()
#%%
num_samples = 1000

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_len = x_train.shape[1]
x_width = x_train.shape[2]
x_train = x_train.reshape((len(x_train)*x_len * x_width))[:num_samples*784]
x_test = x_test.reshape((len(x_test)*x_len * x_width))[:num_samples*784]

y_train = y_train.reshape((len(y_train)*10))[:num_samples*10]
y_test = y_test.reshape((len(y_test)*10))[:num_samples*10]

train_x_file = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "x_train.csv")
test_x_file = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "x_test.csv")
train_y_file = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "y_train.csv")
test_y_file = os.path.join("c:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "y_test.csv")
#%%
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
x_train.to_csv(train_x_file, header = False, index = False)
x_test.to_csv(test_x_file, header = False, index = False)
y_train.to_csv(train_y_file, header = False, index = False)
y_test.to_csv(test_y_file, header = False, index = False)