import torch
from torch.utils.data import Dataset#, DataLoader
from torch import nn
# from torch.nn import functional as F

class dataset(Dataset):

    def __init__(self,x,y):
        # self.x = torch.from_numpy(x)
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]  

    def __len__(self):
        return self.length


class Net(nn.Module):

    def __init__(self,input_shape):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(input_shape,512) # skus aj 512
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)  
        self.fc4 = nn.Linear(128,1)  

        self.dropout = nn.Dropout(p=0.15)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
    

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = torch.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))

        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
