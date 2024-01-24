import os

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split

from model import CnnModel

data = []
labels = []
classes = 43

current_path = os.getcwd()

for i in range(classes):
    path = os.path.join(current_path, 'GSTRB/Train', str(i))
    images = os.listdir(path)
    for a in images:
        image = cv2.imread(os.path.join(path, a), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (30, 30))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = np.transpose(X_test, (0, 3, 1, 2))
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = torchdata.TensorDataset(X_train, y_train)
val_dataset = torchdata.TensorDataset(X_test, y_test)

train_dataloader = torchdata.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torchdata.DataLoader(val_dataset, batch_size=32, shuffle=True)

model = CnnModel(input_channels=3, num_classes=classes)

trainer = L.Trainer(max_epochs=15, log_every_n_steps=1)
trainer.fit(model, train_dataloader, val_dataloader)

# Test the model on GSTRB/Test data
test_csv = pd.read_csv(os.path.join(current_path, 'GSTRB/Test.csv'))
test_labels = test_csv["ClassId"].values
test_imgs = test_csv["Path"].values
test_data = []
for img in test_imgs:
    image = cv2.imread(os.path.join(current_path, 'GSTRB', img), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (30, 30))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test_data.append(image)

test_data = np.array(test_data)
test_data = np.transpose(test_data, (0, 3, 1, 2))
test_data = torch.tensor(test_data, dtype=torch.float32)

test_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

test_dataset = torchdata.TensorDataset(test_data, test_labels)
test_dataloader = torchdata.DataLoader(test_dataset, batch_size=32, shuffle=False)
trainer.test(ckpt_path="best", dataloaders=test_dataloader)
