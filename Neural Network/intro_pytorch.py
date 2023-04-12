import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):

    custom_transform= transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set=datasets.FashionMNIST('./data', train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)

    
    if training == True:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    elif training == False:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
    else:
        loader = "train set = True, test set = False"

    return loader


def build_model():
    model = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 128), # 28x28 = 784 (input), 128 (output)
                    nn.ReLU(),
                    nn.Linear(128, 64), # 128 (input), 64 (output)
                    nn.ReLU(),
                    nn.Linear(64, 10) # 64 (input), 10 (output)
                    )
    return model


def train_model(model, train_loader, criterion, T):
    for epoch in range(T):
        # Set the model to train mode
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        
        # Initialize the loss and accuracy for this epoch
        epoch_loss = 0.0
        epoch_correct = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the epoch loss and accuracy
            epoch_loss += loss.item() * images.shape[0]
            _, predictions = torch.max(outputs.data, 1)
            epoch_correct += (predictions == labels).sum().item()
        
        # Print the training status for this epoch
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = 100.0 * epoch_correct / len(train_loader.dataset)
        print('Train Epoch: {} Accuracy: {}/{} ({:.2f}%) Loss: {:.3f}'.format(
            epoch, epoch_correct, len(train_loader.dataset), epoch_acc, epoch_loss))
    

def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * images.shape[0]
            _, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)

    if show_loss:
        print('Average loss: {:.4f}'.format(test_loss))
    print('Accuracy: {:.2f}%'.format(test_acc))
    

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                    ,'Sneaker','Bag','Ankle Boot']
    
    model.eval()
    with torch.no_grad():
        output = model(test_images[index])
        prob = F.softmax(output, dim=1)
        values, indices = prob.topk(3)
        for i in range(3):
            print('{}: {:.2f}%'.format(class_names[indices[0][i]], values[0][i]*100))