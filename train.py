import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import wandb

from models import SimpleConvNet, MiniVGG, WideResNet, mobilenet_v2
from pytorchtools import EarlyStopping

BATCH_SIZE=32
VALIDATION_FRACTION = 0.2

def create_data_loaders(config):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(VALIDATION_FRACTION * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE,
                                            sampler=train_sampler,
                                            num_workers=2)

    valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=BATCH_SIZE,
                                                sampler=valid_sampler,
                                                num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    config["batch_size"] = BATCH_SIZE
    config["validation_fraction"] = VALIDATION_FRACTION
    config["data_augmentation"] = str(transform)

    return train_loader, valid_loader, test_loader


def create_model(config):
    model_type = config["model_type"] 
    if model_type == "SimpleConvNet":
        if model_type not in config:
            config[model_type] = {"conv1_size" :  32,
                                "conv2_size" :  64,
                                "fc_size" : 128}
        model = SimpleConvNet(**config[model_type])
    elif model_type == "MiniVGG":
        if model_type not in config:
            config[model_type] = {"conv1_size" :  128,
                            "conv2_size" :  256,
                            "classifier_size" : 1024}
        model = MiniVGG(**config[model_type])
    elif model_type == "WideResNet":
        if model_type not in config:
            config[model_type] = {"depth" :  34,
                            "num_classes" :  10,
                            "widen_factor" : 10}
        model = WideResNet(**config[model_type])
    # elif model_type == "ShuffleNetv2":
    #     if model_type not in config:
    #         config[model_type] = {}
    #     model = shufflenet_v2_x0_5()
    elif model_type == "MobileNetv2":
        if model_type not in config:
            config[model_type] = {"pretrained" : False}
        model = mobilenet_v2(num_classes=10, pretrained=config[model_type]["pretrained"])
    else:
        print(f"Error: MODEL_TYPE {model_type} unknown.")
        exit()

    config["num_parameters"] = sum(p.numel() for p in model.parameters())
    config["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model

def setup_training(model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config["use_lr_decay"] = True
    config["lr_decay_rate"] = 0.96

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config["lr_decay_rate"])
    return optimizer, criterion, lr_scheduler


def setup_wandb(model, config):
    print(config)
    wandb.login()
    wandb.init(project="fashion_mnist", anonymous="allow", config=config)
    wandb.watch(model, log="all")

def test(model, test_loader, compute_confusion_matrix=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    correct = 0
    total = 0
    confusion_matrix = torch.zeros(10, 10)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if compute_confusion_matrix:
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
    wandb.log({
      "Test Loss": 100 * correct / total})        
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    if compute_confusion_matrix:
        print(confusion_matrix)




def train(model, optimizer, criterion, lr_scheduler, train_loader, valid_loader, test_loader, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=5, verbose=True)
    valid_losses = []

    print(f"Number of mini-batches: {len(train_loader)} for batch_size {BATCH_SIZE}")
    for epoch in range(20):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                test(model, test_loader)

        if config["use_lr_decay"]:
            print(f"Decreasing learning rate to {lr_scheduler.get_lr()}, i.e. {config['lr_decay_rate']**(epoch+1)*100}%")
            lr_scheduler.step()
        torch.save(model.state_dict(), f"model_epoch{epoch}.h5")
        wandb.save(f"model_epoch{epoch}.h5")  

        model.eval()
        for data in valid_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            valid_losses.append(loss.item())


        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print('Finished training.')
    test(model, test_loader, compute_confusion_matrix=True)

def main():
    print("Starting training...")
    # Train a vanilla CNN
    MODEL_TYPE = "SimpleConvNet"
    config = {"model_type": MODEL_TYPE,
              MODEL_TYPE:  {"conv1_size":  64,
                            "conv2_size":  128,
                            "fc_size": 512}}
    print(f"Training {config}.")
    train_loader, valid_loader, test_loader = create_data_loaders(config)
    model = create_model(config)
    setup_wandb(model, config)
    optimizer, criterion, lr_scheduler = setup_training(model, config)

    train(model, optimizer, criterion, lr_scheduler, train_loader, valid_loader, test_loader, config)

    # Train MobileNetv2 from scratch
    MODEL_TYPE = "MobileNetv2"
    config = {"model_type": MODEL_TYPE,
              MODEL_TYPE : {
                  "pretrained" : False
              }}
    print(f"Training {config}.")
    train_loader, valid_loader, test_loader = create_data_loaders(config)
    model = create_model(config)
    setup_wandb(model, config)
    optimizer, criterion, lr_scheduler = setup_training(model, config)

    train(model, optimizer, criterion, lr_scheduler, train_loader, valid_loader, test_loader, config)
    

if __name__ == "__main__":
    main()
