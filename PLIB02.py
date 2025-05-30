
""" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

""" @@@@@                This is Library of Project functions                     @@@@@ """

""" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""


"""
Function list:
    
1-get_dataset()

2-EEGNet()

3-train_model()

4-predict_sample()

5-find_anchors()

6-extract_features()

7-create_pool()

8-guess_label()

9-check_label()

"""
# select dataset
filename='data_aa.mat'

"""" ************************************************************************************ """
""""     Read Dataset and  Return labeked data(x,y) and Unlabed data(z)                   """
""" ************************************************************************************ """

from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
import mne

# set parameters
sfreq=100
l_freq=8
h_freq=30

def get_dataset(filename):
    # read data
    d=loadmat(filename)
    
    # get data
    dr=np.array(d['data1'])# right
    df=np.array(d['data2'])# foot
    dn=np.array(d['data3'])# not defined
    sampels=dr.shape[2]+df.shape[2]
    # ceate train data
    Train_data=np.empty((400,118,sampels),dtype=float,order='C')
    # merge data1+data2->Train and validation data
    Train_data[:,:, 0:dr.shape[2] ]=dr[:,:, 0:dr.shape[2]]
    Train_data[:,:,dr.shape[2]:sampels ]=df[:,:, 0:df.shape[2]]    
    # create Train labels 
    Train_labels=np.zeros(sampels)
    Train_labels[0:dr.shape[2] ]=0    # class right
    Train_labels[dr.shape[2]:sampels]=1   # class foot
    # x=mne.filter.filter_data(Train_data, sfreq, l_freq, h_freq, picks=None,
    #                           filter_length='auto', l_trans_bandwidth='auto',
    #                           h_trans_bandwidth='auto', n_jobs=1, method='fir',
    #                           iir_params=None, copy=True, phase='zero',
    #                           fir_window='hamming', fir_design='firwin',
    #                           pad='reflect_limited', verbose=None) 
    x=Train_data
    x = np.transpose(x, (2, 1, 0))
    y=Train_labels	
    z = np.transpose(dn, (2, 1, 0))
    
    # Standardize each channel across the samples and time points (x)
    mean = np.mean(x, axis=(0, 2), keepdims=True)
    std = np.std(x, axis=(0, 2), keepdims=True)
    x = (x - mean) / std
    
    # Normalize each channel across the samples and time points to range [0, 1] (x)
    x_min = np.min(x, axis=(0, 2), keepdims=True)
    x_max = np.max(x, axis=(0, 2), keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    
    # Standardize each channel across the samples and time points (z)
    mean = np.mean(z, axis=(0, 2), keepdims=True)
    std = np.std(z, axis=(0, 2), keepdims=True)
    z = (z - mean) / std
    
    # Normalize each channel across the samples and time points to range [0, 1] (z)
    z_min = np.min(z, axis=(0, 2), keepdims=True)
    z_max = np.max(z, axis=(0, 2), keepdims=True)
    z = (z - z_min) / (z_max - z_min)
    
    return(x,y,z)

def get_dataset2(filename):
    # read data
    d=loadmat(filename)
    
    # get data
    dr=np.array(d['data1'])# right
    df=np.array(d['data2'])# foot
    dn=np.array(d['data3'])# not defined
    # ceate train data
    Train_data=np.empty((118,168),dtype=float,order='C')
    # merge data1+data2->Train and validation data
    Train_data[:, 0:80 ]=dr[:, 0:80]
    Train_data[:,80:168]=df[:, 0:88]    
    # create Train labels 
    Train_labels=np.zeros(168)
    Train_labels[0:80  ]=0    # class right
    Train_labels[80:168]=1   # class foot
    # x=mne.filter.filter_data(Train_data, sfreq, l_freq, h_freq, picks=None,
    #                           filter_length='auto', l_trans_bandwidth='auto',
    #                           h_trans_bandwidth='auto', n_jobs=1, method='fir',
    #                           iir_params=None, copy=True, phase='zero',
    #                           fir_window='hamming', fir_design='firwin',
    #                           pad='reflect_limited', verbose=None) 
    x=Train_data
    x = np.transpose(x,   (1,0))
    y=Train_labels	
    z = np.transpose(dn, (1,0))
    
    # Standardize each channel across the samples and time points (x)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    x = (x - mean) / std
    
    # Normalize each channel across the samples and time points to range [0, 1] (x)
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    
    # Standardize each channel across the samples and time points (z)
    mean = np.mean(z, axis=0, keepdims=True)
    std = np.std(z, axis=0, keepdims=True)
    z = (z - mean) / std
    
    # Normalize each channel across the samples and time points to range [0, 1] (z)
    z_min = np.min(z, axis=0, keepdims=True)
    z_max = np.max(z, axis=0, keepdims=True)
    z = (z - z_min) / (z_max - z_min)
    
    return(x,y,z)

def get_dataset2(filename):
    # read data
    d=loadmat(filename)
    
    # get data
    dr=np.array(d['data1'])# right
    df=np.array(d['data2'])# foot
    
    # ceate train data
    Train_data=np.empty((59,200),dtype=float,order='C')
    # merge data1+data2->Train and validation data
    Train_data[:, 0:100 ]=dr[:, 0:100]
    Train_data[:,100:200]=df[:, 0:100]    
    # create Train labels 
    Train_labels=np.zeros(200)
    Train_labels[0:100  ]=0    # class right
    Train_labels[100:200]=1   # class foot
    
    x=Train_data
    x = np.transpose(x, (1,0))
    y=Train_labels	
   
    # Standardize each channel across the samples and time points (x)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    x = (x - mean) / std
    
    # Normalize each channel across the samples and time points to range [0, 1] (x)
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    
    return(x,y)
""" ************************************************************************************ """
""""                  EEGNet architecture for Train model                                """ 
"""************************************************************************************  """

import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

def EEGNet(nb_classes, Chans=118, Samples=400, dropoutRate=0.50, kernLength=50,
           F1=8, D=2, F2=16, norm_rate=0.25):
    
    input1   = Input(shape=(Chans, Samples, 1))

    # First temporal convolutional layer
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                  input_shape = (Chans, Samples, 1),
                                  use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = tf.keras.constraints.max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = Dropout(dropoutRate)(block1)
    
    # Separable convolutional layer
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    # Flattening and output layer
    flatten      = Flatten(name = 'flatten')(block2)
    dense        = Dense(nb_classes, name = 'dense',
                         kernel_constraint = tf.keras.constraints.max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

from tensorflow.keras import layers, models
def build_eegnet(input_shape, num_classes):
    model = models.Sequential()

    # First Conv Layer (1D Conv)
    model.add(layers.Conv2D(8, (1, 4), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Second Conv Layer (1D Conv)
    model.add(layers.Conv2D(16, (1, 4), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Depthwise Separable Conv Layer (for EEG data)
    model.add(layers.DepthwiseConv2D((1, 4), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.25))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layer (Dense)
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU())
    model.add(layers.Dropout(0.5))

    # Output Layer (Softmax activation for classification)
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

""" ************************************************************************************ """
"""                 Train main Knowlage model and prediction                             """ 
"""************************************************************************************  """

from keras.utils import to_categorical
from matplotlib import pyplot as plt
import keras
import numpy as np
from keras.models import Sequential,Model
from tensorflow.keras.models import Model
from sklearn import metrics
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold

def train_model2(filename=filename,pre_trian_model='trained_model.h5'):
    x, y, z =get_dataset(filename=filename)
    X_train=x
    Y_train=y
    kf=KFold(5,shuffle=True,random_state=10)    
    fold=0
    e_acc=[]
    t_acc=[]
    for train, test in kf.split(X_train,Y_train):
        fold+=1
        print('###################################################################')
        print(f"fold #{fold}")
        x_train=X_train[train]
        y_train=Y_train[train]
        x_test=X_train[test]
        y_test=Y_train[test]
        
        ###############################################################################
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        # One-hot encode the labels
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
 
        ######################## model definition and training #########################        
        eegnet_model = EEGNet(nb_classes=2, Chans=118, Samples=400)
        eegnet_model.compile(loss='binary_crossentropy', 
                             optimizer=Adam(learning_rate=0.00001), 
                             metrics=['accuracy'])

        history = eegnet_model.fit(x_train, y_train, batch_size=64, epochs=10,verbose=2,
                                       validation_split=0.2)
        ################################################################################
        test_loss, evaluate_accuracy = eegnet_model.evaluate(x_train, y_train)
        print(f"\n Model Evalution accuracy: {evaluate_accuracy * 100:.2f}%")        
        test_loss, test_accuracy = eegnet_model.evaluate(x_test, y_test)
        print(f"\n Model Test accuracy: {test_accuracy * 100:.2f}%")        
        e_acc.append(evaluate_accuracy)
        t_acc.append(test_accuracy)
    evaluate_accuracy=sum(e_acc)/len(e_acc)    
    evaluate_accuracy=sum(t_acc)/len(t_acc)
    return eegnet_model,evaluate_accuracy,evaluate_accuracy
    

def train_model(filename=filename,pre_trian_model='trained_model.h5'):
    x, y, z =get_dataset(filename=filename)
	
	# Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=20,
                                                                        random_state=42)   
    # get copy of dataset
    x_train, x_test, y_train, y_test=X_train, X_test, Y_train, Y_test
   
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Step 4: Compile the model
    eegnet_model = EEGNet(nb_classes=2, Chans=118, Samples=400)
    eegnet_model.compile(loss='binary_crossentropy', 
                         optimizer=Adam(learning_rate=0.00001), 
                         metrics=['accuracy'])

    history = eegnet_model.fit(x_train, y_train, batch_size=16, epochs=10,verbose=2,
                                   validation_split=0.2)
    # Step 6: Evaluate the model on the test set
    test_loss, evaluate_accuracy = eegnet_model.evaluate(x_train, y_train)
    print(f"\n Model Evalution accuracy: {evaluate_accuracy * 100:.2f}%")
    
    test_loss, test_accuracy = eegnet_model.evaluate(x_test, y_test)
    print(f"\n Model Test accuracy: {test_accuracy * 100:.2f}%")
    

    print('accuracy and loss of training and validation plott')
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print('Saveing model...')  
    eegnet_model.save('trained_model.h5')
    print(eegnet_model.summary())
    return eegnet_model,evaluate_accuracy,test_accuracy

def train_model_featured(filename=filename,pre_trian_model='trained_model.h5'):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    x, y, z =get_dataset(filename=filename)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20,
                                                                        random_state=42)   
    # get copy of dataset
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    x_train_csp = csp.fit_transform(x_train,y_train)
    x_test_csp = csp.transform(x_test)

    # Standardize the features before clustering
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_csp)
    
    # Assume you already have your EEG data loaded in 'x' and 'y' arrays
    # x has shape (168, 4) and y has shape (168,)
    
    # Normalize EEG data
    x_train_csp = x_train_csp / np.max(np.abs(x_train_csp), axis=1, keepdims=True)  # Normalize across the channels
    x_test_csp = x_test_csp / np.max(np.abs(x_test_csp), axis=1, keepdims=True)  # Normalize across the channels
    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)  
    # Reshape the data to match EEGNet's input expectations (samples, time points, channels, 1)
    x_train_csp = np.expand_dims(x_train_csp, axis=-1)  # Shape becomes (train_samples, time_points, channels, 1)
    x_test_csp = np.expand_dims(x_test_csp, axis=-1)    # Shape becomes (test_samples, time_points, channels, 1)
    
    # Define EEGNet model
    def build_eegnet(input_shape, num_classes):
        model = models.Sequential()
    
        # First Conv Layer (1D Conv)
        model.add(layers.Conv2D(8, (1, 4), padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.25))
    
        # Second Conv Layer (1D Conv)
        model.add(layers.Conv2D(16, (1, 4), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.25))
    
        # Depthwise Separable Conv Layer (for EEG data)
        model.add(layers.DepthwiseConv2D((1, 4), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.25))
    
        # Global Average Pooling
        model.add(layers.GlobalAveragePooling2D())
    
        # Fully Connected Layer (Dense)
        model.add(layers.Dense(64))
        model.add(layers.BatchNormalization())
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.5))
    
        # Output Layer (Softmax activation for classification)
        model.add(layers.Dense(num_classes, activation='softmax'))

    return model

    # Build the model
    input_shape = (x_train_csp.shape[1], x_train_csp.shape[2], 1)  # (time_points, channels, 1)
    num_classes = y_train.shape[1]  # Number of classes in y
    
    model = build_eegnet(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(x_train_csp, y_train, epochs=20, batch_size=32,
                                          validation_data=(x_test_csp, y_test))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

    
    print('Saveing model...')  
    eegnet_model.save('trained_model.h5')
    print(eegnet_model.summary())
    return eegnet_model

def train_model_augmented(filename=filename,pre_trian_model='trained_model.h5'):
    # load Augmented x,y from EEG dataset
    x=np.load('x_augmented.npy')
    y=np.load('y_augmented.npy')
    
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=20,
                                                                        random_state=42)   
    # get copy of dataset
    x_train, x_test, y_train, y_test=X_train, X_test, Y_train, Y_test
   
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Step 4: Compile the model
    eegnet_model = EEGNet(nb_classes=2, Chans=118, Samples=400)
    eegnet_model.compile(loss='binary_crossentropy', 
                         optimizer=Adam(learning_rate=0.00001), 
                         metrics=['accuracy'])

    history = eegnet_model.fit(x_train, y_train, batch_size=8, epochs=10,verbose=2,
                                   validation_split=0.2)
    # Step 6: Evaluate the model on the test set
    test_loss, evaluate_accuracy = eegnet_model.evaluate(x_train, y_train)
    print(f"\n Model Evalution accuracy: {evaluate_accuracy * 100:.2f}%")
    
    test_loss, test_accuracy = eegnet_model.evaluate(x_test, y_test)
    print(f"\n Model Test accuracy: {test_accuracy * 100:.2f}%")
    
    print('accuracy and loss of training and validation plott')
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    print('Saveing model...')  
    eegnet_model.save('trained_model.h5')
    print(eegnet_model.summary())
    return eegnet_model,evaluate_accuracy,test_accuracy

def predict_sample(selected_sample,pre_trian_model='trained_model.h5'):
    model=keras.models.load_model('trained_model.h5')
    selected_sample = selected_sample.reshape(selected_sample.shape[0], selected_sample.shape[1], selected_sample.shape[2], 1)
    pred=model.predict(selected_sample)  
    return pred


""" ************************************************************************************ """
"""           Select Query Anchores from Labeled data from dataset                       """ 
"""************************************************************************************  """

from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import mne
from mne.decoding import CSP

#Query Anchor selection
def find_anchors(filename=filename):
    
    # read data
    d=loadmat(filename)
    # get data
    d1=np.array(d['data1'])# right
    d2=np.array(d['data2'])# foot
    d3=np.array(d['data3'])# not defined

    # reshape data
    dr=d1.reshape(118,400,80)
    df=d2.reshape(118,400,88)
    dn=d3.reshape(118,400,112) 


    # create Train,validation and Test array 
    # dr=56+16+8=88 : dr=Train+Validation+Test
    # df=62+17+9=88 : df=Train+Validation+Test
    Train_data=np.empty((118,400,151),dtype=float,order='C')
    Test_data=np.empty((118,400,17),dtype=float,order='C')#17=8+9

    # merge data1+data2->Train and validation data
    Train_data[:,:, 0:72 ]=dr[:,:, 0:72]
    Train_data[:,:,72:151]=df[:,:, 0:79]

    # merge: data1+data2->Test_data
    Test_data[:,:, 0:8 ]=dr[:,:, 72:80]
    Test_data[:,:, 8:17]=df[:,:, 79:88]


    # create Train and validation labels 
    Train_labels=np.zeros(151)
    # set labels for 118 trails
    Train_labels[0:72  ]=0    # class right
    Train_labels[72:151]=1   # class foot
     
    # create Test labels 
    Test_labels=np.zeros(17)
    # set labels for 17 trails
    Test_labels[0:8 ]=0    # class right
    Test_labels[8:17]=1    # class foot

    # set parameters
    sfreq=100
    l_freq=8
    h_freq=30

    # dataset(X,Y)
    x_train=mne.filter.filter_data(Train_data, sfreq, l_freq, h_freq, picks=None,
                                  filter_length='auto', l_trans_bandwidth='auto',
                                  h_trans_bandwidth='auto', n_jobs=1, method='fir',
                                  iir_params=None, copy=True, phase='zero',
                                  fir_window='hamming', fir_design='firwin',
                                  pad='reflect_limited', verbose=None)
    x_train=x_train.reshape(151,118,400)
    y_train=Train_labels

    x_test=mne.filter.filter_data(Test_data, sfreq, l_freq, h_freq, picks=None,
                                  filter_length='auto', l_trans_bandwidth='auto',
                                  h_trans_bandwidth='auto', n_jobs=1, method='fir',
                                  iir_params=None, copy=True, phase='zero',
                                  fir_window='hamming', fir_design='firwin',
                                  pad='reflect_limited', verbose=None)
    x_test=x_test.reshape(17,118,400)
    y_test=Test_labels

    print('\n filtering done...\n')
    # feature extraction using CSP method 
    print('featue extraction using CSP method....\n')
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    x_train_csp = csp.fit_transform(x_train,y_train)
    x_test_csp = csp.transform(x_test)

    # Standardize the features before clustering
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_csp)

    ######################## clustering using k-means ############################

    print('Clustering trials using K-means method...\n')
    # Create a K-means model with 10 clusters
    kmeans = KMeans(n_clusters=2,random_state=42, verbose=1)

    # Fit the model to the training data
    kmeans.fit(x_scaled)

    # Get the cluster labels for the training data
    y_pred = kmeans.predict(x_scaled)

    # Convert the cluster labels to categorical format
    y_pred = to_categorical(y_pred)

    ####################### most representative anchor selection ##################

    # Get the cluster labels and centroids
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Function to find the closest anchors (trials closest to the centroids)

    def find_closest_anchors(X, centroids):
        anchors = []
        for centroid in centroids:
            # Compute distances between the centroid and all points (both X and centroid should be 2D)
            distances = np.linalg.norm(X - centroid, axis=1)  # X is X_scaled (2D), centroid is 1D
            # Find the index of the closest point
            anchor_index = np.argmin(distances)
            anchors.append(X[anchor_index])
        return np.array(anchors)

    # Get the most representative anchors
    anchors = find_closest_anchors(x_scaled, centroids)
    
    # Output the selected anchors
    print(f'Representative anchors from the two clusters: \n{anchors}')

    # save query anchors
    np.savez_compressed('Query-anchores.npz',anchors)
    # find index of anchors
    print("Finding the index of anchors....\n")
    index = np.where((x_scaled == anchors[0]).all(axis=1))[0]

    # Display the result
    if index.size > 0:
        print(f'Index of anchor class(0) is:  {index[0]}')
    index = np.where((x_scaled == anchors[1]).all(axis=1))[0]

    # Display the result
    if index.size > 0:
        print(f'Index of anchor class(1) is : {index[0]}')
    return anchors




""" ************************************************************************************ """
"""                 Extract Features of unlabeled samples                                """
"""************************************************************************************  """

from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler

def extract_features(filename=filename):
    x,y,z=get_dataset(filename)
    
    x_u=z # nuseen data with no label
	
    print('\n filtering done...\n')
    
    # feature extraction using CSP method 
    print('featue extraction using CSP method....\n')
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    x_train_csp = csp.fit_transform(x_train,y_train)
    x_u_csp = csp.transform(x_u)
	
    # Standardize the features 
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_u_csp)
    return x_scaled 




""" ************************************************************************************ """
"""               create pool of unlabeled-feature extracted samples                     """
"""************************************************************************************  """

def create_pool(filename=filename):
    x_scaled=extract_features(filename=filename)
    print('featue extraction using CSP method....\n')
    return x_scaled

""" ************************************************************************************ """
"""         stimates label for a selected unlabeled-feature extracted sample             """
""" ************************************************************************************ """
from tensorflow.keras.losses import CosineSimilarity

def guess_label1(selected_sample, x_scaled, anchors):
    
    anchor01=anchors[0,:] # class 0 anchor
    anchor02=anchors[1,:] # class 1 anchor
    
    # Reshape anchors to match the shape of a for broadcasting
    anchor01 = anchor01.reshape(1,4)
    anchor02 = anchor02.reshape(1,4) 

    # Create a TensorFlow cosine similarity function
    cosine_similarity = CosineSimilarity(axis=1)

    similarity_score01=[]
    similarity_score02=[]
    
    # Compute the cosine similarity 
    for i in range(168):   
        a = selected_sample.reshape(1,4) # get x_scaled[i]    
        similarity_score01.append((cosine_similarity(a, anchor01)).numpy()+1)
        similarity_score02.append((cosine_similarity(a, anchor02)).numpy()+1)
        
    # Convert to positive similarity score (because CosineSimilarity returns negative similarity by default)

    # Print the result
    print("Cosine Similarity score for class 0:\n", similarity_score01[0])
   
    print("Cosine Similarity score for class 1:\n", similarity_score02[0])
   
    if similarity_score01 > similarity_score02:
        estimated_label=0
    else:
        estimated_label=1
    return estimated_label



""" ************************************************************************************ """
"""   Checks the stimated label for a selected unlabeled-feature extracted sample        """                                                                     
""" ************************************************************************************ """

def check_label1(selected_sample,estimated_label, eegnet_model):
    pred=predict_sample(selected_sample, eegnet_model)  
    predicted_class = np.argmax(pred, axis=1) # output 0 for class0 ,output 1 for class1
    
    if predicted_class==estimated_label:
        return  1
    else:
        return -1
    
  
  
  
