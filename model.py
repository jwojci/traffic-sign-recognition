import lightning as L
import torch.nn as nn
import torch.optim


class CnnModel(L.LightningModule):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=0)
        self.norm2d1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), 'same')
        self.norm2d2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(64, 64, (2, 2), (1, 1), 'same')
        self.norm2d3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(576, 256)
        self.relu4 = nn.ReLU(True)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2d1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2d2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.norm2d3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log("train_accuracy", accuracy.item() * 100, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        self.log("val_accuracy", accuracy.item() * 100, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(labels == predictions).item() / (len(labels) * 1.0)

        self.log("test_accuracy", accuracy * 100, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
