import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms,datasets
from user.model import user_model
import logging
import time

def fit(model,device,train_loader,val_loader,optimizer,criterion,epochs,logger):

  data_loader = {'train':train_loader,'val':val_loader}
  print("Fitting the model...")
  train_loss,val_loss=[],[]
  train_acc,val_acc=[],[]
  for epoch in range(epochs):
    loss_per_epoch,val_loss_per_epoch=0,0
    acc_per_epoch,val_acc_per_epoch,total,val_total=0,0,0,0
    for phase in ('train','val'):
      for i,data in enumerate(data_loader[phase]):
        inputs,labels  = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        #preding classes of one batch
        preds = torch.max(outputs,1)[1]
        #calculating loss on the output of one batch
        loss = criterion(outputs,labels)
        if phase == 'train':
          acc_per_epoch+=(labels==preds).sum().item()
          total+= labels.size(0)
          optimizer.zero_grad()
          #grad calc w.r.t Loss func
          loss.backward()
          #update weights
          optimizer.step()
          loss_per_epoch+=loss.item()
        else:
          val_acc_per_epoch+=(labels==preds).sum().item()
          val_total+=labels.size(0)
          val_loss_per_epoch+=loss.item()
    logger.info("Epoch: {} Loss: {:0.6f} Acc: {:0.6f} Val_Loss: {:0.6f} Val_Acc: {:0.6f}".format(epoch+1,loss_per_epoch/len(train_loader),acc_per_epoch/total,val_loss_per_epoch/len(val_loader),val_acc_per_epoch/val_total))
    train_loss.append(loss_per_epoch/len(train_loader))
    val_loss.append(val_loss_per_epoch/len(val_loader))
    train_acc.append(acc_per_epoch/total)
    val_acc.append(val_acc_per_epoch/val_total)
  return train_loss,val_loss,train_acc,val_acc

def user_train(train_loader=None , val_loader=None , model_path=None, device="cpu",logger=None,epoch=5):
  np.random.seed(42)
  torch.manual_seed(42)
 
  model = user_model().to(device)
  optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-6)
  criterion = nn.CrossEntropyLoss()
  train_loss,val_loss,train_acc,val_acc = fit(model,device,train_loader,val_loader,optimizer,criterion,epoch,logger)
  torch.save(model.state_dict(),model_path)

if __name__=='__main__':
  logfile =  "logs/log_" + time.strftime('%Y-%m-%d-%H-%M-%S' , time.localtime(time.time())) + ".txt"
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)
  handler = logging.FileHandler(logfile,"a")
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logger.addHandler(handler)
  logger.addHandler(console)

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  dataset = datasets.MNIST(root = './data', train=True, transform=transform, download=True)
  train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])

  use_cuda=True
  device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

  train_loader = torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_set,batch_size=128,shuffle=True)
  model_path = './models/mnist_model.pt'
  train(train_loader=train_loader , val_loader=val_loader, model_path=model_path, device=device,logger=logger)
  
