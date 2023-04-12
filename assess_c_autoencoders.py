# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:36:20 2023

@author: TOM3O
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


handle = os.path.join("C:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "results.csv")
handle_optimized = os.path.join("C:/", "Users", "TOM3O", "Desktop", "Work", "ENGG4540", "AutoEncoder", "results_optimized.csv")

df = pd.read_csv(handle, names = ["time", "training_mse", "test_mse"])
df_optimized = pd.read_csv(handle_optimized, names = ["time", "training_mse", "test_mse"])


plt.plot(df["training_mse"], label = 'train')
plt.plot(df["test_mse"], label = 'test')
plt.legend()
plt.title("C Implementation Autoencoder Training History")
plt.show()
plt.close()

plt.plot(df_optimized["training_mse"], label = 'train')
plt.plot(df_optimized["test_mse"], label = 'test')
plt.legend()
plt.title("Optimized C Implementation Autoencoder Training History")
plt.show()
plt.close()