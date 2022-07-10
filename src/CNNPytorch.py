# Import Libraries
import time
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from ConvNeuralNet import ConvNeuralNet
from ModelInterface import ModelInterface
from torchinfo import summary


class CNNPytorch(ModelInterface):
    def __init__(self, train_ds_path, validation_ds_path, batch_size, image_size, epochs, model_save_path
                 , model_layers, model_linear_layers
                 ):
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.EPOCHS = epochs
        self.train_ds_path = train_ds_path
        self.validation_ds_path = validation_ds_path
        self.model_save_path = model_save_path
        self.train_dataset = 0
        self.test_dataset = 0
        self.train_dl = 0
        self.test_dl = 0
        self.class_names = 0
        self.data_augmentation = 0
        self.num_classes = 2
        self.model = 0
        self.scores = 0
        self.device = "cpu"
        self.criterion = 0
        self.optimizer = 0
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.val_TP = 0
        self.val_FP = 0
        self.val_TN = 0
        self.val_FN = 0
        self.val_history = 0
        self.model_layers = model_layers
        self.model_linear_layers = model_linear_layers
        self.history = [[0 for x in range(10)] for y in range(self.EPOCHS)]

    def read_dataset(self):
        train_transformation = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        test_transformation = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        # Load train, validation and test dataset
        train_file = self.train_ds_path
        test_file = self.validation_ds_path

        self.train_dataset = ImageFolder(train_file, train_transformation)
        self.test_dataset = ImageFolder(test_file, test_transformation)

        # PyTorch data loaders
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_dl = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE)
        self.class_names = self.train_dl.dataset.classes

    def compile_model(self):
        self.model = ConvNeuralNet(self.model_layers, self.model_linear_layers)
        # self.model = ConvNeuralNet3Layers()
        # Set Loss function with criterion
        self.criterion = nn.BCELoss()
        # Set optimizer with optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def compute_measurements(self, loc_TP, loc_FP, loc_TN, loc_FN):
        # Acuratete = TP+TN/(TP+TN+FP+FN) -> toate corecte din toate
        # Precision = TP / (TP+FP) -> true positiv din toate pozitivele prezise
        # Recall = TP / P -> true positiv din toate pozitivele reale
        # Sensitivity = Recall
        # Specificity = TN / (TN + FP) -> true negativ din toate negativele reale
        accuracy = ((loc_TP + loc_TN) / (loc_TP + loc_TN + loc_FP + loc_FN)) * 100
        if loc_TP + loc_FP > 0:
            precision = loc_TP / (loc_TP + loc_FP)
        else:
            precision = 0
        if (loc_TP + loc_FN) > 0:
            recall = loc_TP / (loc_TP + loc_FN)
        else:
            recall = 0
        if (loc_TN + loc_FP) > 0:
            specificity = loc_TN / (loc_TN + loc_FP)
        else:
            specificity = 0
        return accuracy, precision, recall, specificity

    def update_confusionmatrix(self, y_actual, y_hat, loc_TP, loc_FP, loc_TN, loc_FN):
        # print(y_hat)
        # print(y_actual)
        for i in range(len(y_hat)):
            if y_actual[i].item() > 0.5 and y_hat[i].item() == 1:
                loc_TP += 1
            if y_hat[i].item() == 0 and y_actual[i].item() > 0.5:
                loc_FP += 1
            if y_actual[i].item() <= 0.5 and y_hat[i].item() == 0:
                loc_TN += 1
            if y_hat[i].item() == 1 and y_actual[i].item() <= 0.5:
                loc_FN += 1
        return loc_TP, loc_FP, loc_TN, loc_FN

    def train(self):
        # PyTorch - Training the Model
        for e in range(self.EPOCHS):
            self.model.train()
            # define the loss value after the epoch
            losss = 0.0
            number_of_sub_epoch = 0
            start = time.time()
            # loop for every training batch (one epoch)
            for images, labels in self.train_dl:
                # in pytorch you have assign the zero for gradien in any sub epoch
                self.optimizer.zero_grad()
                # create the output from the network
                out = self.model(images)
                _, predicted = out.max(1)
                # count the loss function
                loss = self.criterion(out, labels.unsqueeze(1).float())
                # count the backpropagation
                loss.backward()
                # learning
                self.optimizer.step()
                # add new value to the main loss
                losss += loss.item()
                number_of_sub_epoch += 1
                self.TP, self.FP, self.TN, self.FN = self.update_confusionmatrix(out, labels, self.TP, self.FP, self.TN,
                                                                                 self.FN)
            end = time.time()
            print("Epoch {}: Loss: {} Time: {} ".format(e, losss / number_of_sub_epoch, (end - start)))
            print("TP: {} FP: {} TN: {} FN: {}".format(self.TP, self.FP, self.TN, self.FN))
            accuracy, precision, recall, specificity = self.compute_measurements(self.TP, self.FP, self.TN, self.FN)

            print("Accuracy: {} Precision: {} Recall: {} Specificity: {} ".format(accuracy,
                                                                                  precision,
                                                                                  recall,
                                                                                  specificity))
            self.history[e][0] = accuracy
            self.history[e][1] = precision
            self.history[e][2] = recall
            self.history[e][3] = specificity
            self.history[e][4] = losss / number_of_sub_epoch

            self.TP = 0
            self.FP = 0
            self.TN = 0
            self.FN = 0
            self.validate(e)

        return self.history

    def save_model(self):
        torch.save(self.model, self.model_save_path)

    def load_model(self):
        self.model = torch.load(('E:/automatica/master_an2/Disertatie/Breast_xray/breast_pytorch_CNN.pth'))
        self.criterion = nn.BCELoss()
        # Set optimizer with optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def validate(self, e):
        # PyTorch - Comparing the Results
        total = 0
        val_loss = 0.0
        val_losss = 0.0
        number_of_sub_epoch = 0
        self.model.eval()
        # self.optimizer.zero_grad()
        with torch.no_grad():
            for images, labels in self.test_dl:
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                val_loss = self.criterion(outputs, labels.unsqueeze(1).float())
                number_of_sub_epoch += 1
                total += labels.size(0)
                val_losss += val_loss.item()
                # print(outputs)
                self.val_TP, self.val_FP, self.val_TN, self.val_FN = self.update_confusionmatrix(outputs, labels,
                                                                                                 self.val_TP,
                                                                                                 self.val_FP,
                                                                                                 self.val_TN,
                                                                                                 self.val_FN)
        accuracy, precision, recall, specificity = self.compute_measurements(self.val_TP, self.val_FP, self.val_TN,
                                                                             self.val_FN)
        print('On {} test images: Val_Loss {}% with PyTorch'.format(total, val_losss / number_of_sub_epoch))
        print("Validation TP: {} FP: {} TN: {} FN: {}".format(self.val_TP, self.val_FP, self.val_TN, self.val_FN))
        print("Val_Accuracy: {} Val_Precision: {} Val_Recall: {} Val_Specificity: {} ".format(accuracy,
                                                                                              precision,
                                                                                              recall,
                                                                                              specificity))
        self.history[e][5] = accuracy
        self.history[e][6] = precision
        self.history[e][7] = recall
        self.history[e][8] = specificity
        self.history[e][9] = val_losss / number_of_sub_epoch

        self.val_TP = 0
        self.val_FP = 0
        self.val_TN = 0
        self.val_FN = 0
