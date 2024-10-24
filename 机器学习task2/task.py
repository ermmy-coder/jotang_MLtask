import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 32
#作答：即批量大小，是一次处理的数据量大小。
#过大：速度快但模型精度下降。
#过小：速度慢。
learning_rate = 3
#作答：即学习率，可以抽象地理解为每次调整权重时的程度，走到最佳权重点每一步地步长。
#学习率过低：调整幅度小。
#学习率过高：容易偏离目标。
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor()
])

# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#`DataLoader()`:用于数据加载，自动批量处理、打乱数据等
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        #`nn.Conv2d()`:创建二维卷积类
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

        pass

    def forward(self, x):
        # TODO:这里补全你的前向传播
        #`sigmoid`激活
        x=torch.sigmoid(self.conv1(x))
        #`max_pool2d()`:池化
        x=self.pool(x)
        #`torch.flatten()`:展平为一维向量
        x=torch.sigmoid(self.conv2(x))
        x=self.pool(x)
        x=torch.flatten(x,1)
        x=torch.sigmoid(self.fc1(x))
        x=torch.sigmoid(self.fc2(x))
        x=self.fc3(x)
        return x
        pass

# TODO:补全
model = Network()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), learning_rate)  

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()
