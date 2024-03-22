import torch
import sys, os
import argparse
import numpy as np
import seaborn as sns
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import time
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report



sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')

from augmentation import Aug
from XModel import XModel
from loader import session
import optparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Default
cession='g' # GPU runtime
epoch = 1
dir_path = ""
batch_size = 32
lr=0.0001
weight_decay=0.0000001

parser = optparse.OptionParser("Train XModel model.")
parser.add_option("-e", "--epoch", type=int, dest='epoch', help='Number of epochs used for training the X model.')
parser.add_option("-v", "--version", dest='version', help='Version 0.1.')
parser.add_option("-s", "--cession", type="string",dest='session', help='Training session. Use g for GPU, t for TPU.')
parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
parser.add_option("-l", "--rate",  type=float, dest='rate', help='Learning rate.')
parser.add_option("-w", "--decay", type=float, dest='decay', help='Weight decay.')
parser.add_option("-p", "--plot", action="store_true", dest='plot', help='Plot training and validation metrics.')
# parser.add_option("-a", "--auc", action="store_true", dest='auc', help='Calculate and plot AUC.')
# parser.add_option("-c", "--confmatrix", action="store_true", dest='confmatrix', help='Calculate and plot confusion matrix.')

(options,args) = parser.parse_args()

plot_metrics = False
# calculate_auc = False
# calculate_confmatrix = False

if options.session:
    cession = options.session
if options.dir==None:
    print (parser.usage)
    exit(0)
else:
    dir_path = options.dir
if options.batch:
    batch_size = int(options.batch)
if options.epoch:
    epoch = int(options.epoch)
if options.rate:
    lr = float(options.rate)
if options.decay:
    weight_decay = float(options.decay)
if options.plot:
    plot_metrics = True
# if options.auc:
#     calculate_auc = True
# if options.confmatrix:
#     calculate_confmatrix = True


if cession=='t':
    print('USING TPU.')
    device = xm.xla_device()

batch_size, dataloaders, dataset_sizes = session(cession, dir_path, batch_size)

#X model definition
model = XModel(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024, dropout_rate=0.1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
num_epochs = epoch
min_val_loss=10000
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

def train_tpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics):

    model_path = 'weight/xmodel_deepfake_sample_1.pth'
    if os.path.exists(model_path):
        print("Loading saved model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_loss']
    else:
        print("Train from begining...")
        start_epoch = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    epoch_loss = None
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            phase_idx=0

            epoch_loss = running_loss / dataset_sizes[phase]

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #break
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        xm.optimizer_step(optimizer, barrier=True)
                        xm.mark_step()

                if phase_idx%100==0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, phase_idx * batch_size, dataset_sizes[phase],\
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss, min_loss))
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    with open('weight/xmodel_deepfake_sample_1.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    if epoch_loss is not None:
        state = {'epoch': num_epochs+1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'min_loss': epoch_loss}
        torch.save(state, 'weight/xmodel_deepfake_sample_1.pth')

    # if calculate_auc == True:
    #     auc_score = calculate_auc(model, dataloaders, dataset_sizes)
    #     print('AUC:', auc_score)

    # if calculate_confmatrix == True:
    #     cm = calculate_confusion_matrix(model, dataloaders, dataset_sizes)
    #     print(cm)

    if plot_metrics == True:
        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(val_loss, label="val")
        plt.plot(train_loss, label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/training_validation_loss.png')  # Save the plot as a file
        plt.close()

        # Plotting the training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy")
        plt.plot([acc.cpu().numpy() for acc in val_accu], label="val")
        plt.plot([acc.cpu().numpy() for acc in train_accu], label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/training_validation_accuracy.png')  # Save the plot as a file
        plt.close()

    return train_loss,train_accu,val_loss,val_accu, min_loss

def train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics):
    model_path = 'weight/xmodel_deepfake_sample_1.pth'
    if os.path.exists(model_path):
        print("Loading saved model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_loss']
    else:
        start_epoch = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    epoch_loss = None
    total_epochs = start_epoch + num_epochs

    for epoch in range(start_epoch, total_epochs):
        print('Epoch {}/{}'.format(epoch, total_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            phase_idx=0

            epoch_loss = running_loss / dataset_sizes[phase]

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #break
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() # GPU || CPU

                if phase_idx%100==0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, phase_idx * batch_size, dataset_sizes[phase], \
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss, min_loss))
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    with open('weight/xmodel_deepfake_sample_1.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    if epoch_loss is not None:
        state = {'epoch': num_epochs+1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'min_loss': epoch_loss}
        torch.save(state, 'weight/xmodel_deepfake_sample_1.pth')

    test(model, dataloaders, dataset_sizes)

    # if calculate_auc == True:
    auc_score = calculate_auc(model, dataloaders, dataset_sizes)
    print('AUC:', auc_score)

    # if calculate_confmatrix == True:
    cm = calculate_confusion_matrix(model, dataloaders, dataset_sizes)
    print('confusion_matrix', cm)

    if plot_metrics == True:
        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(val_loss, label="val")
        plt.plot(train_loss, label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('result/training_validation_loss.png')  # Save the plot as a file
        plt.close()

        # Plotting the training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Accuracy")
        plt.plot([acc.cpu().numpy() for acc in val_accu], label="val")
        plt.plot([acc.cpu().numpy() for acc in train_accu], label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('result/training_validation_accuracy.png')  # Save the plot as a file
        plt.close()

    return train_loss,train_accu,val_loss,val_accu, min_loss

def calculate_auc(model, dataloaders, dataset_sizes):
    model.eval()
    all_labels = []
    all_preds = []

    # Iterate over test data
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        # Store probabilities and true labels for ROC calculation
        outputs = outputs.softmax(dim=1)  # Convert to probabilities
        all_labels.extend(labels.tolist())
        all_preds.extend(outputs[:, 1].tolist())  # Assuming binary classification

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('result/roc_curve.png')  # Save the plot as a file
    plt.close()

    return roc_auc

def calculate_confusion_matrix(model, dataloaders, dataset_sizes):
    model.eval()
    all_labels = []
    all_preds = []

    # Iterate over test data
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.tolist())
        all_preds.extend(predictions.tolist())

    # Calculate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    # F1 Score, Precision, and Recall
    f1_score = report['weighted avg']['f1-score']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    print(f"F1 Score: {f1_score:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Format and plot confusion matrix
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.yticks(np.arange(2), ['Fake', 'Real'], size = 16)
    plt.title('Confusion Matrix')
    plt.savefig('result/confusion_matrix.png')
    plt.close()

    return cm

def test(model, dataloaders, dataset_sizes):
    model.eval()
    correct_predictions = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)
        correct_predictions += (predictions == labels).sum().item()

    accuracy = (correct_predictions / dataset_sizes['test']) * 100
    print('Prediction accuracy: ', accuracy, '%')

if cession == 't':
    train_tpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics)
else:
    train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, plot_metrics)
