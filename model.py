import torch.nn as nn
import torch.nn.functional as F
import torch
class user_model(nn.Module):
  def __init__(self):
    super(user_model, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 32, 3, 1)
    self.conv3= nn.Conv2d(32, 64, 3, 1)
    self.conv4= nn.Conv2d(64, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(1024, 200)
    self.fc2 = nn.Linear(200, 200)
    self.fc3 = nn.Linear(200, 10)

  def forward(self, x):
    x = self.conv1(x)   
    x = F.relu(x)
    x = self.conv2(x)   
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  
    x = self.conv3(x)       
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)  
    x = torch.flatten(x,1)  
    x = self.fc1(x)         
    x = F.relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)        
    return x
