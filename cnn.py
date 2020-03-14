

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

trainset = torchvision.datasets.MNIST('../mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)) 
                             ])) 
    
testset = torchvision.datasets.MNIST("../mnist_data/", 
                             download=True,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, ),(0.3081, ))
                           ]))
    
trainloader = torch.utils.data.DataLoader(trainset,                                         
                                         shuffle=True)


testloader = torch.utils.data.DataLoader(testset,                                         
                                         shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256,256)
        self.fc = nn.Linear(256,128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
     
        x = self.fc7(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00060, momentum=0.9)



for epoch in range(10):  
    running_loss = 0.0
    for i, data in enumerate(trainloader): 
        inputs, labels = data
        optimizer.zero_grad()
   
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 600 == 0:    
            print('epoch : %d 진행도 : %.d%% loss : %.3f' %
                  (epoch + 1, i/600, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0

    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += 1
      correct += (predicted == labels).sum().item()

      print('%d / %d \n정확도: %lf %%' %(correct,total,
    100.0 * correct / total))
print('끝')


    
