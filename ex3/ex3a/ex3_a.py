# Name: Ofir Cohen
# ID: 312255847
# Date: 15/5/2020

# from trains import Task
# task = Task.init(project_name="Deep_Learning_Frameworks_Playground", task_name="babysitting_learning_process")
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from one_fc import OneFC
from two_fc import TwoFC
from cnn_one import cnn_one_bn
from cnn_two import cnn_two_bn
from vgg_16 import vgg16_bn

# from trains import Task
# task = Task.init(project_name="Babysitting_learning_process", task_name="Babysitting_learning_process")
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fit_predict(model, epochs=2, lr=1e-3):
    CUDA = torch.cuda.is_available()
    trainset, trainloader, testset, testloader, classes = load_cifar_10_data_tf()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model['model'].parameters(), lr)
    train_loss_values = []
    accuracy_values = []
    
    print("Starting training model: {}".format(model['name']))
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get images and labels
            images, labels = data
            # zero gradient params
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model['model'](images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i %  2000 == 1999:
                print("[{}, {}] loss: {}".format(epoch+1, i+1, running_loss/2000))
                train_loss_values.append(running_loss / len(trainset))
                running_loss = 0.0
    
    print("Finished training model: {}".format(model['name']))
    print("====================================================")
    print("Starting testing model: {}".format(model['name']))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model['model'](images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy_values.append(correct / total)
    
    print("Accuracy of the network on the test images: {}".format(100 * correct / total))
    plt.plot(train_loss_values)
    plt.title("model {} loss".format(model['name']))
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train'], loc='upper left')
    plt.savefig('./plots/{}_loss_plot.png'.format(model['name']))
    plt.clf()
    plt.plot(accuracy_values)
    plt.title("model {} accuracy".format(model['name']))
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['test'], loc='upper left')
    plt.savefig('./plots/{}_accuracy_plot.png'.format(model['name']))
    plt.clf()
    print("====================================================")
    
''' End function '''


def load_cifar_10_data_tf():
	torch.manual_seed(1)
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	
	return trainset, trainloader, testset, testloader, classes


def main():
    # Initialize models
    nn_models = []   
    one_fc_model = OneFC()
    two_fc_model = TwoFC()
    one_conv_one_fc = cnn_one_bn()
    two_conv_two_fc = cnn_two_bn()
    vgg16 = vgg16_bn()

    nn_models.append({ 'name': 'ONE_FC', 'model': one_fc_model })
    nn_models.append({ 'name': 'TWO_FC', 'model': two_fc_model })
    nn_models.append({ 'name': 'ONE_CNN', 'model': one_conv_one_fc })
    nn_models.append({ 'name': 'TWO_CNN', 'model': two_conv_two_fc })
    nn_models.append({ 'name': 'VGG16', 'model': vgg16 })

    # Execute learning
    for model in nn_models:
        fit_predict(model)

if __name__ == "__main__":
    main()