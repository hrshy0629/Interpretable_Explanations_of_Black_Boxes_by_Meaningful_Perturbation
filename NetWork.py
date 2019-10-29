# -*- coding: utf-8 -*-
from Logging import logger
import torch.nn as nn
import torch
from utils import cifar10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Test_nn(object):
    def __init__(self, test_classifier, learning_rate, num_epoch):
        self.classifier = test_classifier
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
    def add_data(self, batch_size):
        train_loader, test_loader = cifar10(batch_size)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        #print(type(self.test_loader))
    def train_test(self):
        #training
        model = self.classifier
        criterion = nn.CrossEntropyLoss()
        optims = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epoch):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optims.zero_grad()
                loss.backward()
                optims.step()

                if (i+1)%100 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.num_epoch}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4}")

        #testing
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            logger.info('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        logger.info('Save model')
        torch.save(model.state_dict(), 'model_save/model.ckpt')

