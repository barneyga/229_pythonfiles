import torchvision.models as models
import torch
import torchvision
import time
from torchvision import transforms
import copy
from torchvision import datasets
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import MoviesDataset
from torch.utils.data import DataLoader
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
import warnings
# from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
from functools import reduce
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    writer = SummaryWriter()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    targetnames = ['War', 'Fantasy', 'Mystery', 'TV Movie', 'Science Fiction', 'Western', 'Comedy', 'Documentary', 'Crime', 'Action', 'Music', 'Adventure', 'Family', 'Thriller', 'History', 'Horror', 'Foreign', 'Drama', 'Romance', 'Animation']

    epoch_num = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        preds_training_epoch, labels_training_epoch = torch.Tensor([]).cpu(), torch.Tensor([]).cpu()
        preds_dev_epoch, labels_dev_epoch = torch.Tensor([]).cpu(), torch.Tensor([]).cpu()

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())

                    preds = (outputs >= 0.5)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.byte())
                if phase == 'train':
                    writer.add_scalar('Train/Loss', loss, epoch_num)
                    preds_training_epoch = torch.cat((preds_training_epoch, preds.cpu().float()))
                    labels_training_epoch = torch.cat((labels_training_epoch, labels.data.byte().cpu().float()))
                elif phase == 'dev':
                    writer.add_scalar('Val/Loss', loss, epoch_num)
                    preds_dev_epoch = torch.cat((preds_dev_epoch, preds.cpu().float()))
                    labels_dev_epoch = torch.cat((labels_dev_epoch, labels.data.byte().cpu().float()))




            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * 20)
            epoch_one_match = 0
            epoch_all_match = 0
            preds_training_epoch, preds_dev_epoch = preds_training_epoch.numpy(), preds_dev_epoch.numpy()
            labels_training_epoch, labels_dev_epoch = labels_training_epoch.numpy(), labels_dev_epoch.numpy()
            if phase == 'train':
                epoch_one_match = (np.sum(np.all(preds_training_epoch - labels_training_epoch <= 0, axis=1)) - np.sum(np.all(preds_training_epoch== 0, axis=1))) / len(labels_training_epoch)
                epoch_all_match = np.sum(np.all(preds_training_epoch == labels_training_epoch, axis=1)) / len(labels_training_epoch)     
                print(classification_report(labels_training_epoch, preds_training_epoch, target_names=targetnames))
            elif phase == 'dev':
                epoch_one_match = (np.sum(np.all(preds_dev_epoch - labels_dev_epoch <= 0, axis=1)) - np.sum(np.all(preds_dev_epoch == 0, axis=1))) / len(labels_dev_epoch)
                epoch_all_match = np.sum(np.all(preds_dev_epoch == labels_dev_epoch, axis=1)) / len(labels_dev_epoch)     
                print(classification_report(labels_dev_epoch, preds_dev_epoch, target_names=targetnames))
            preds_training_epoch, preds_dev_epoch = torch.from_numpy(preds_training_epoch), torch.from_numpy(preds_dev_epoch)
            labels_training_epoch, labels_dev_epoch = torch.from_numpy(labels_training_epoch), torch.from_numpy(labels_dev_epoch)
            print('{} Loss: {:.4f} Acc: {:.4f} Hamming Loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, (1-epoch_acc)))
            print('One Match: {:.4f} All Match: {:.4f}'.format(epoch_one_match, epoch_all_match))
            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'dev':
                val_acc_history.append(epoch_acc)

        print()
        epoch_num += 1
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best dev Acc: {:4f}'.format(best_acc))

    writer.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    torch.set_printoptions(threshold=5000)
    warnings.filterwarnings('ignore')

    # torch.cuda.empty_cache()
    torch.multiprocessing.freeze_support()
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    # data_dir = "./data/sets"

    model_name = 'resnet'

    num_classes = 20

    batch_size = 100
    num_epochs = 20

    feature_extract = False

    # Initialize the model for this run
    model_ft, input_size = initialize_model('resnet', num_classes, feature_extract, use_pretrained=True)
    # model_ft = torch.load('ResNetPickled.pt')

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'dev': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'dev']}
    # print(image_datasets)
    DATA_PATH = ''
    TRAIN_DATA = 'train224x224'
    DEV_DATA = 'dev224x224'
    TRAIN_IMG_FILE = 'train_files.txt'
    DEV_IMG_FILE = 'dev_files.txt'
    TRAIN_LABEL_FILE = 'train_labels.txt'
    DEV_LABEL_FILE = 'dev_labels.txt'
    dset_train = MoviesDataset.DatasetProcessing(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, data_transforms['train'])
    dset_dev = MoviesDataset.DatasetProcessing(DATA_PATH, DEV_DATA, DEV_IMG_FILE, DEV_LABEL_FILE, data_transforms['dev'])
    image_datasets = {'train': dset_train, 'dev': dset_dev}

    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dset_dev, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders_dict = {'train': train_loader, 'dev': dev_loader}

    model_ft = model_ft.cuda()



    params_to_update = model_ft.parameters()

    # Observe that all parameters are being optimized
    # optimizer_ft = torch.optim.Adam(params_to_update, lr=0.1, weight_decay=0.01)
    # optimizer_ft = torch.optim.Adam(params_to_update, weight_decay=0.01)
    optimizer_ft = torch.optim.Adam(params_to_update)

    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MultiLabelSoftMarginLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    torch.save(model_ft, 'ResNetPickled.pt')