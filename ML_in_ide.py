import torch 
from torchvision import datasets, transforms, models
#import matplotlib.pyplot as plt
#import seaborn as sb
from torch import nn, optim
import pickle as p
#import torch.nn.functional as f

#label mapping.
f=open('set.dat','wb')
Dict={1:'cat', 2:'Fish'}
p.dump(Dict,f)
f.close()

#label maping.
g=open('set.dat','rb')
try:
  #while True:
  E=p.load(g)
    #print(E)
except:
  g.close()

#loading the datasets
#data='images'
train_path="C:/Users/Srijito Ghosh/.vscode/Programs_in_vscode/train"
test_path="C:/Users/Srijito Ghosh/.vscode/Programs_in_vscode/test"
val_path="C:/Users/Srijito Ghosh/.vscode/Programs_in_vscode/val"

#loading the dataloaders
mea=[0.456,0.489,0.457] #the mean.
st=[0.226,0.225,0.239] #the standard deviations.
train_transforms=transforms.Compose([transforms.RandomRotation(90),
                                     transforms.RandomHorizontalFlip(),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mea,
                                                        std=st)])
transforms=transforms.Compose([transforms.CenterCrop(224),
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mea,
                                                      std=st)])
#creating datasets with ImageFolder. ImageFolder is a class which takes arguments of
#paths and transforms that are to be applied to the images passed into the model.

train_data = datasets.ImageFolder(train_path, train_transforms)
test_data = datasets.ImageFolder(test_path, transforms)
val_data = datasets.ImageFolder(val_path, transforms)

train_loaders=torch.utils.data.DataLoader(train_data, 32, shuffle = True)
test_loaders=torch.utils.data.DataLoader(test_data, 32, shuffle = True)
val_loaders=torch.utils.data.DataLoader(val_data, 32, shuffle = True)

#creating the model class.
class CNN_Net(nn.Module):
  
  def __init__(self):
    super(CNN_Net, self).__init__()
    self.features=nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=2, padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2),
                                nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3), stride=2, padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3,stride=2),
                                nn.Conv2d(in_channels=20, out_channels=35, kernel_size=3, stride=2),
                                nn.ReLU())
                                #nn.MaxPool2d(kernel_size=3, stride=2))
    self.avgpool=nn.AdaptiveAvgPool2d((6,6))
    
    self.classifier=nn.Sequential(nn.Linear(35 * 6 * 6, 20),
                                  nn.ReLU(),
                                  nn.Linear(20,10),
                                  nn.ReLU(),
                                  nn.Linear(10,2),
                                  nn.LogSoftmax(dim=1))
    
  def forward(self,x):
    x=self.features(x)
    x=self.avgpool(x)
    x=torch.flatten(x, 1)
    x=self.classifier(x)
    return x
  
if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

criterion=nn.NLLLoss()

#creating an instance of the model
model=CNN_Net()
#creating the optimizer
optimizer= optim.Adam(model.parameters(), 0.001) #first parameter: part of the model that needs to be get the weights updated. second parameter: the learning rate.

def validation():
  val_loss=0
  model.to(device)
  model.eval()
  for images, labels in val_loaders:
    images,labels=images.to(device), labels.to(device)
    log_ps=model.forward(images)
    loss=criterion(log_ps)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    val_loss+=loss.item()
    ps=torch.exp()
    equals=(labels.data==ps.max)
    val_accuracy = equals.type(equals.FloatTensor).mean().item()
    return val_loss, val_accuracy

#training loop
def train():
  train_loss=0
  train_accuracy=0
  model.to(device)
  model.train()
  print_every=32
  steps=0
  epochs=int(input('Enter number of epochs.'))
  for images,labels in train_loaders:
    images,labels=images.to(device),labels.to(device)
    log_ps=model.forward(images) #log probabilities.
    loss=criterion(log_ps, labels)#calculating the loss.
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    train_loss+=loss.item()
    ps=torch.exp(log_ps)
    equals=(labels.data == ps.max(dim=1)[1])
    train_accuracy+=equals.type_as(torch.FloatTensor()).mean().item()
    steps+=1

    if steps % print_every == 0:
      model.eval()
      with torch.no_grad():
        val_loss, val_accuracy= validation()
      #printing out the accuracies is not a necessary thing but rather a really nice practise. 
      print('Training accuracy: ', train_accuracy)
      print('Training loss: ', train_loss)
      print('Validation loss: ', val_loss)
      print('Validation accuracy: ', val_loss)
      model.train()

train()

#test loop
def test():
  model.eval()
  test_loss=0
  model.to(device)
  test_accuracy = 0
  for images, labels in test_loaders:
    images, labels = images.to(device), labels.to(device)
    log_ps=model.forward(images)
    loss=criterion(log_ps, labels)
    
    ps=torch.exp(log_ps)
    equals=(labels.data == ps.max(dim=1))
    print(type(equals))
    print(int(equals))
    #test_accuracy+=equals.type(torch.FloatTensor).mean().items()
  print("Test accuracy: ", test_accuracy)
test()

#saving the model.
def save_model():
  torch.save(model.state_dict, 'Checkpoint1.pth') #a .pth file is a file that stores the model's parameter, state and weights in a pickled format.
save_model()

def load_the_model():
  model = CNN_Net()
  model_state_dict = torch.load('checkpoint.pth')
  model.load_state_dict(model_state_dict)
#load_the_model()

def prediction():
    model=load_the_model()
#image processing for making an inference.
#def image_process_andInference():
