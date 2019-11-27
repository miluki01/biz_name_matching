import torch
import torch.nn as nn
from torchsummary import summary

class NameMatchingModel(nn.Module):
    def __init__(self):
        super(NameMatchingModel, self).__init__()

        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NameMatchingModel().to(device)
    summary(model, (2, 9))
