import torchvision.models as models
import torch
import time
from torchvision import transforms
from torchvision import datasets
import os
import matplotlib.pyplot as plt
import numpy as np
import MoviesDataset
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

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

    batch_size = 80
    # num_epochs = 50

    feature_extract = False

    # Initialize the model for this run
    model = torch.load('ResNetPickled34_noHT.pt')
    model.eval()
    # model_ft = torch.load('ResNetPickled.pt')

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    # print(image_datasets)
    DATA_PATH = ''
    TRAIN_DATA = 'train224x224'
    TEST_DATA = 'test224x224'
    TRAIN_IMG_FILE = 'train_files.txt'
    TEST_IMG_FILE = 'test_files.txt'
    TRAIN_LABEL_FILE = 'train_labels.txt'
    TEST_LABEL_FILE = 'test_labels.txt'
    targetnames = ['War', 'Fantasy', 'Mystery', 'TV Movie', 'Science Fiction', 'Western', 'Comedy', 'Documentary', 'Crime', 'Action', 'Music', 'Adventure', 'Family', 'Thriller', 'History', 'Horror', 'Foreign', 'Drama', 'Romance', 'Animation']

    dset_train = MoviesDataset.DatasetProcessing(DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, data_transforms['train'])
    dset_test = MoviesDataset.DatasetProcessing(DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, data_transforms['test'])
    image_datasets = {'train': dset_train, 'dev': dset_test}

    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'dev': test_loader}

    optimizer = torch.optim.Adam(model.parameters())

    preds_dev_epoch, labels_dev_epoch = torch.Tensor([]).cpu(), torch.Tensor([]).cpu()

    running_loss = 0.0
    running_corrects = 0

    class_names = targetnames

    # Iterate over data.
    for inputs, labels in dataloaders['dev']:
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels.float())
        preds = (outputs >= 0.5)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data.byte())

        preds_dev_epoch = torch.cat((preds_dev_epoch, preds.cpu().float()))
        labels_dev_epoch = torch.cat((labels_dev_epoch, labels.data.byte().cpu().float()))

    epoch_loss = running_loss / len(dataloaders['dev'].dataset)
    epoch_acc = running_corrects.double() / (len(dataloaders['dev'].dataset) * 20)
    epoch_one_match = 0
    epoch_all_match = 0
    preds_dev_epoch = preds_dev_epoch.numpy()
    labels_dev_epoch = labels_dev_epoch.numpy()
    epoch_one_match = (np.sum(np.all(preds_dev_epoch - labels_dev_epoch <= 0, axis=1)) - np.sum(np.all(preds_dev_epoch == 0, axis=1))) / len(labels_dev_epoch)
    epoch_all_match = np.sum(np.all(preds_dev_epoch == labels_dev_epoch, axis=1)) / len(labels_dev_epoch)     
    print(classification_report(labels_dev_epoch, preds_dev_epoch, target_names=targetnames))
    preds_dev_epoch = torch.from_numpy(preds_dev_epoch)
    labels_dev_epoch = torch.from_numpy(labels_dev_epoch)
    print('{} Loss: {:.4f} Acc: {:.4f} Hamming Loss: {:.4f}'.format('test', epoch_loss, epoch_acc, (1-epoch_acc)))
    print('One Match: {:.4f} All Match: {:.4f}'.format(epoch_one_match, epoch_all_match))

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['dev']):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)
                preds = (outputs >= 0.5)
                identified_genres_list = []
                for twenty in labels:
                    class_indices = np.where(twenty.cpu().numpy() == 1)
                    l = []
                    if 0 in class_indices[0]:
                        l.append('War')
                    if 1 in class_indices[0]:
                        l.append('Fantasy')
                    if 2 in class_indices[0]:
                        l.append('Mystery')
                    if 3 in class_indices[0]:
                        l.append('TV Movie')
                    if 4 in class_indices[0]:
                        l.append('Science Fiction')
                    if 5 in class_indices[0]:
                        l.append('Western')
                    if 6 in class_indices[0]:
                        l.append('Comedy')
                    if 7 in class_indices[0]:
                        l.append('Documentary')
                    if 8 in class_indices[0]:
                        l.append('Crime')
                    if 9 in class_indices[0]:
                        l.append('Action')
                    if 10 in class_indices[0]:
                        l.append('Music')
                    if 11 in class_indices[0]:
                        l.append('Adventure')
                    if 12 in class_indices[0]:
                        l.append('Family')
                    if 13 in class_indices[0]:
                        l.append('Thriller')
                    if 14 in class_indices[0]:
                        l.append('History')
                    if 15 in class_indices[0]:
                        l.append('Horror')
                    if 16 in class_indices[0]:
                        l.append('Foreign')
                    if 17 in class_indices[0]:
                        l.append('Drama')
                    if 18 in class_indices[0]:
                        l.append('Romance')
                    if 19 in class_indices[0]:
                        l.append('Animation')
                    identified_genres_list.append(l)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('Ground Truth: {}'.format(identified_genres_list[j]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
    visualize_model(model)
    plt.ioff()
    plt.show()