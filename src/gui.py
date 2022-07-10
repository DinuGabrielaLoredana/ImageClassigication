import os
from tkinter import filedialog
from tkinter import *
import tkinter as tk

import numpy
from PIL import ImageTk, Image
import torch
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Rescaling, Flatten
from keras.models import load_model, Sequential
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import tensorflow as tf

from CNNKeras import CNNKeras
from CNNPytorch import CNNPytorch
from ConvNeuralNet import ConvNeuralNet


class Gui:
    def __init__(self):
        self.classes = {0: 'normal', 1: 'cancer'}
        self.top = tk.Tk()
        canvas = Canvas(
            self.top,
            height=1024,
            width=1024,
            bg="#fff"
        )
        canvas.place(x=500, y=0)

        self.backgroundcolor = "#3399ff"
        canvas.create_rectangle(
            0, 0, 1024, 1024,
            outline="#fb0",
            fill=self.backgroundcolor)
        self.label = Label(self.top, font=('arial', 15, 'bold'))
        self.labelAccuracy = Label(self.top, font=('arial', 15, 'bold'))
        self.radioVar = StringVar(self.top, "1")
        self.breast_image = Label(self.top)
        self.filepath = ""
        self.EPHOCS = 5
        self.ephocs_textbox = Entry(self.top)
        self.batch_size = 64
        self.batch_size_textbox = Entry(self.top)
        self.number_layers = 10
        self.number_layers_textbox = Entry(self.top)
        self.cnn_current_layer = ""
        self.cnn_current_layer_textbox = Entry(self.top)
        self.radiovar_cnn_current_layer = StringVar(self.top, "1")
        self.cnn_in_channel = Entry(self.top)
        self.cnn_out_channel = Entry(self.top)
        self.cnn_kernel_size = Entry(self.top)
        self.maxpool_kernel_size = Entry(self.top)
        self.maxpool_stride = Entry(self.top)
        self.dropout = Entry(self.top)
        self.linear_in_channel = Entry(self.top)
        self.linear_out_channel = Entry(self.top)
        self.model_keras = Sequential()
        self.model_keras.add(Rescaling(1. / 255, input_shape=(50, 50, 3)))
        self.model_pytorch_layers = torch.nn.ModuleList()
        self.model_pytorch_linear_layers = torch.nn.ModuleList()
        self.train_ds_path = Entry(self.top)
        self.val_ds_path = Entry(self.top)
        self.save_path = Entry(self.top)
        self.labelResults = Label(self.top, font=('arial', 10, 'bold'))
        self.labelResults.place(x=515,y=415)
        self.init_gui()

    # initialize GUI
    def init_gui(self):
        self.top.geometry('1024x1024')
        self.top.title('Image classification')
        self.top.configure(background='#FFFFFF')
        upload = Button(self.top, text="Upload an image", command=self.upload_image, padx=10, pady=5)
        # Dictionary to create multiple buttons
        values = {"Pytorch": "1",
                  "Keras": "2"
                  }
        upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
        upload.place(x=100, y=500)
        self.breast_image.place(x=20, y=200)
        self.label.place(x=100, y=150)
        self.labelAccuracy.place(x=200, y=150)
        # Loop is used to create multiple Radio buttons rather than creating each button separately
        i = 0
        for (text, value) in values.items():
            i += 1
            Radiobutton(self.top, text=text, font=('arial', 10, 'bold'), variable=self.radioVar, background='#FFFFFF',
                        value=value).place(x=100, y=50 + 25 * i)
        tk.Button(self.top, text="Load model", command=self.open_file, background='#364156', foreground='white', font=('arial', 10, 'bold')).place(x=100, y=125)
        Label(self.top, text="Number of epochs", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=550, y=75)
        self.ephocs_textbox.place(x=550, y=110)
        Label(self.top, text="Batch size", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=550, y=145)
        self.batch_size_textbox.place(x=550, y=180)
        Label(self.top, text="Training dataset path", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=550, y=205)
        self.train_ds_path.place(x=550, y=240)
        Label(self.top, text="Validation dataset path", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=550, y=275)
        self.val_ds_path.place(x=550, y=310)
        Label(self.top, text="Save model path", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=550, y=345)
        self.save_path.place(x=550, y=380)

        Radiobutton(self.top, text="Conv2D", font=('arial', 10, 'bold'),  variable=self.radiovar_cnn_current_layer, background=self.backgroundcolor,
                    value="1").place(x=800, y=75)
        Label(self.top, text="In channels (only pytorch)", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=100)
        self.cnn_in_channel.place(x=800, y=125)
        Label(self.top, text="Out channels", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=150)
        self.cnn_out_channel.place(x=800, y=175)
        Label(self.top, text="Kernel size", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=200)
        self.cnn_kernel_size.place(x=800, y=225)
        Radiobutton(self.top, text="MaxPool2d", font=('arial', 10, 'bold'), background=self.backgroundcolor, variable=self.radiovar_cnn_current_layer,
                    value="2").place(x=800, y=250)
        Label(self.top, text="Kernel size(only pytorch)", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=275)
        self.maxpool_kernel_size.place(x=800, y=300)
        Label(self.top, text="Stride(only pytorch)", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=325)
        self.maxpool_stride.place(x=800, y=350)
        Radiobutton(self.top, text="Dropout", font=('arial', 10, 'bold'), variable=self.radiovar_cnn_current_layer, background=self.backgroundcolor,
                    value="3").place(x=800, y=375)
        Label(self.top, text="Dropout rate", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=400)
        self.dropout.place(x=800, y=425)
        Radiobutton(self.top, text="Linear", font=('arial', 10, 'bold'), variable=self.radiovar_cnn_current_layer, background=self.backgroundcolor,
                    value="4").place(x=800, y=450)
        Label(self.top, text="In channels(only pytorch)", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=475)
        self.linear_in_channel.place(x=800, y=500)
        Label(self.top, text="Out channels", font=('arial', 10, 'bold'), foreground='white', background=self.backgroundcolor).place(x=800, y=525)
        self.linear_out_channel.place(x=800, y=550)
        Radiobutton(self.top, text="Sigmoid", font=('arial', 10, 'bold'), variable=self.radiovar_cnn_current_layer, background=self.backgroundcolor ,
                    value="5").place(x=800, y=575)
        tk.Button(self.top, text="Add layer", command=self.add_layer, background='#364156', foreground='white', font=('arial', 10, 'bold')).place(x=800, y=600)
        tk.Button(self.top, text="Compile model", command=self.compile_model, background='#364156', foreground='white', font=('arial', 10, 'bold')).place(x=800, y=650)
        tk.Button(self.top, text="Reset model", command=self.reset_model, background='#364156', foreground='white', font=('arial', 10, 'bold')).place(x=800, y=700)

        self.top.mainloop()

    def reset_model(self):
        self.model_keras = Sequential()
        self.model_keras.add(Rescaling(1. / 255, input_shape=(50, 50, 3)))
        self.model_pytorch_layers = torch.nn.ModuleList()
        self.model_pytorch_linear_layers = torch.nn.ModuleList()
        self.labelResults.configure(foreground='#ffffff', background=self.backgroundcolor, text="")

    def add_layer(self):
        layer_type = str(self.radiovar_cnn_current_layer.get())
        if "2" == str(self.radioVar.get()):
            if "1" == layer_type:
                self.model_keras.add(Conv2D(int(self.cnn_out_channel.get()), int(self.cnn_kernel_size.get()),
                                            padding='same', activation='relu'))
            if "2" == layer_type:
                self.model_keras.add(MaxPooling2D())
            if "3" == layer_type:
                self.model_keras.add(Dropout(float(self.dropout.get())))
                self.model_keras.add(Flatten())
            if "4" == layer_type:
                # self.model_keras.add(Dense(int(self.linear_out_channel.get()), activation='relu'))
                print()
            if "5" == layer_type:
                self.model_keras.add(Dense(1, activation='sigmoid'))
        elif "1" == str(self.radioVar.get()):
            if "1" == layer_type:
                self.model_pytorch_layers.append(torch.nn.Conv2d(in_channels=int(self.cnn_in_channel.get()), out_channels=int(self.cnn_out_channel.get()),
                                    kernel_size=int(self.cnn_kernel_size.get())))
                self.model_pytorch_layers.append(torch.nn.ReLU())
            if "2" == layer_type:
                self.model_pytorch_layers.append(torch.nn.MaxPool2d(int(self.maxpool_kernel_size.get()), int(self.maxpool_stride.get())))
            if "3" == layer_type:
                self.model_pytorch_layers.append(torch.nn.Dropout(float(self.dropout.get())))
            if "4" == layer_type:
                self.model_pytorch_linear_layers.append(torch.nn.Linear(int(self.linear_in_channel.get()), int(self.linear_out_channel.get())))
            if "5" == layer_type:
                self.model_pytorch_linear_layers.append(torch.nn.Sigmoid())



    def compile_model(self):
        self.EPHOCS = int(self.ephocs_textbox.get())
        self.batch_size = int(self.batch_size_textbox.get())
        self.labelResults.configure(foreground='#ffffff', background=self.backgroundcolor, text="")
        if "2" == str(self.radioVar.get()):
            algorithm_cnn = CNNKeras(self.train_ds_path.get(),
                                     self.val_ds_path.get(),
                                     self.batch_size,
                                     50,
                                     self.EPHOCS,
                                     "E:/automatica/master_an2/Disertatie/Breast_xray/split/test/models/"+self.save_path.get(),
                                     self.model_keras
                                     )
            algorithm_cnn.read_dataset()
            algorithm_cnn.augment_data()
            algorithm_cnn.compile_model()
            history = algorithm_cnn.train()
            self.labelResults.configure(foreground='#ffffff',
                                        background=self.backgroundcolor,
                                        text="Precision = {:.2f}\n Accuracy = {:.2f}\nRecall = {:.2f}\n"
                                             "Specificity = {:.2f}\nLoss = {:.2f}\n""Validation Precision = {:.2f}\n"
                                             "Validation  Accuracy = {:.2f}\nValidation Recall = {:.2f}\n"
                                             "Validation Specificity = {:.2f}\n"
                                             "Validation Loss = {:.2f}\n".format(float(history.history["precision"][0]*100),
                                                                                 float(history.history["accuracy"][0]*100),
                                                                                 float(history.history["recall"][0]*100),
                                                                                 float(history.history["specificity_at_sensitivity"][0]*100),
                                                                                 float(history.history["loss"][0]*100),
                                                                                 float(history.history["val_precision"][0]*100),
                                                                                 float(history.history["val_accuracy"][0]*100),
                                                                                 float(history.history["val_recall"][0]*100),
                                                                                 float(history.history[
                                                                                           "val_specificity_at_sensitivity"][
                                                                                           0]*100),
                                                                                 float(history.history["val_loss"][0]*100),
                                                                                       )
                                         )
            algorithm_cnn.save_model()
        elif "1" == str(self.radioVar.get()):
            algorithm_cnn = CNNPytorch(self.train_ds_path.get(),
                                       self.val_ds_path.get(),
                                       self.batch_size,
                                       50,
                                       self.EPHOCS,
                                       "E:/automatica/master_an2/Disertatie/Breast_xray/split/test/models/"+self.save_path.get(),
                                       self.model_pytorch_layers,
                                       self.model_pytorch_linear_layers
                                       )
            algorithm_cnn.read_dataset()
            algorithm_cnn.compile_model()
            history = algorithm_cnn.train()
            self.labelResults.configure(foreground='#ffffff',
                                        background=self.backgroundcolor,
                                        text="Precision = {:.2f}\n Accuracy = {:.2f}\nRecall = {:.2f}\n"
                                             "Specificity = {:.2f}\nLoss = {:.2f}\n""Validation Precision = {:.2f}\n"
                                             "Validation  Accuracy = {:.2f}\nValidation Recall = {:.2f}\n"
                                             "Validation Specificity = {:.2f}\n"
                                             "Validation Loss = {:.2f}\n".format(float(history[self.EPHOCS-1][0]),
                                                                                 float(history[self.EPHOCS-1][1]*100),
                                                                                 float(history[self.EPHOCS-1][2]*100),
                                                                                 float(history[self.EPHOCS-1][3]*100),
                                                                                 float(history[self.EPHOCS-1][4]*100),
                                                                                 float(history[self.EPHOCS-1][5]),
                                                                                 float(history[self.EPHOCS-1][6]*100),
                                                                                 float(history[self.EPHOCS-1][7]*100),
                                                                                 float(history[self.EPHOCS-1][8]*100),
                                                                                 float(history[self.EPHOCS-1][9]*100)
                                                                                 )
                                        )
            algorithm_cnn.save_model()

    def open_file(self):
        file = filedialog.askopenfile(mode='r')
        if file:
            self.filepath = os.path.abspath(file.name)

    def classify(self, file_path):

        if "1" == str(self.radioVar.get()):
            # 'E:/automatica/master_an2/Disertatie/Breast_xray/breast_pytorch_CNN.pth'
            model1 = torch.load(self.filepath)
            model1.eval()
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            )
            preprocess = transforms.Compose([
                transforms.Resize(50),
                transforms.ToTensor(),
                normalize
            ])
            img_pil = Image.open(file_path)
            img_tensor = preprocess(img_pil).float()
            img_tensor = img_tensor.unsqueeze_(0)
            fc_out = model1(img_tensor)
            output = fc_out.detach().numpy()
            if 0.5 > float(output[0]):
                sign = 0
                self.label.configure(foreground='#011638', background="#ffffff", text=self.classes[sign])
                self.labelAccuracy.configure(foreground='#011638', background="#ffffff", text=1 - float(output[0]))
            else:
                sign = 1
                self.label.configure(foreground='#011638', background="#ffffff", text=self.classes[sign])
                self.labelAccuracy.configure(foreground='#011638', background="#ffffff", text=float(output[0]))

        elif "2" == str(self.radioVar.get()):
            # 'breast_keras_CNN.h5'
            model = load_model(self.filepath)
            img = tf.keras.utils.load_img(
                file_path, target_size=(50, 50)
            )
            # Create a batch
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            # clasify image
            predictions = model.predict(img_array)
            if 0.5 > predictions[0]:
                sign = self.classes[0]
                self.labelAccuracy.configure(foreground='#011638', background="#ffffff", text=1 - float(predictions[0]))
                self.label.configure(foreground='#011638', background="#ffffff", text=sign)
            else:
                sign = self.classes[1]
                self.labelAccuracy.configure(foreground='#011638', background="#ffffff", text=float(predictions[0]))
                self.label.configure(foreground='#011638', background="#ffffff", text=sign)
        else:
            self.label.configure(foreground='#011638', background="#ffffff", text="no prediction possible")
            print(self.radioVar)

    def show_classify_button(self, file_path):
        classify_b = Button(self.top, text="Classify Image", command=lambda: self.classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
        classify_b.place(x=100, y=550)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            resize_image = uploaded.resize((250, 250))
            uploaded.thumbnail(((self.top.winfo_width() / 2.25), (self.top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(resize_image)
            self.breast_image.configure(image=im, height=250, width=250)
            self.breast_image.image = im
            self.label.configure(text='')
            self.labelAccuracy.configure(text='')
            self.show_classify_button(file_path)
        except:
            pass
