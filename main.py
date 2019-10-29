# -*- coding: utf-8 -*-
from Logging import logger
from model.LeNet import lenet
from model.ResNet import ResNet18
from NetWork import Test_nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.01
kernel = 5
input = 3
num_epochs = 10
def Test_NetWork(classifier, batch_size, learning_rate, num_epochs):
    test_nn = Test_nn(classifier, learning_rate, num_epochs)
    test_nn.add_data(batch_size)
    test_nn.train_test()

if __name__ == '__main__':
    logger.level = 'debug'
    logger.addFileHandler(path='log/Cifar.log')
    model = lenet(input, kernel).to(device)
    #model = ResNet18().to(device)
    Test_NetWork(model, batch_size, learning_rate, num_epochs)
    logger.info(f'The testing process is over!')