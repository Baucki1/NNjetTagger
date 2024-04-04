from typing import List, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as ss
import h5py

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from NNimageMaker import CNNimageMaker
from TreeHolder import TreeHolder

from NNjetTagger import NNjetTagger

class CNN_jet_tagger(NNjetTagger):
    def __init__(self, input_type=True, model_name="CNNjetTagger"): # define all default values at init
        super().__init__(True, model_name)
        
        # overwrite superclass defaults with CNN specific defaults
        self.metrics = [tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"), 
                        tf.keras.metrics.AUC(name="auc"), 
                        tf.keras.metrics.Precision(name="precision"), 
                        tf.keras.metrics.TruePositives(name="true_positives", thresholds=self.threshold), 
                        tf.keras.metrics.TrueNegatives(name="true_negatives", thresholds=self.threshold), 
                        tf.keras.metrics.FalsePositives(name="false_positives", thresholds=self.threshold), 
                        tf.keras.metrics.FalseNegatives(name="false_negatives", thresholds=self.threshold)]
        self.metrics_labels = ["Binary Accuracy", "ROC AUC", "Precision", "True Positives", "True Negatives", "False Positives", "False Negatives"]
        
        self.loss = tf.keras.losses.BinaryCrossentropy()
        
        self.optimizer = tf.keras.optimizers.Adam
        self.learning_rate = 0.001
        
        self.checkpoint_format = ".keras"
        self.checkpoint_path = "models/" + model_name + "/cp-{epoch:04d}" + self.checkpoint_format
        
        callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False, save_freq='epoch', verbose=1)
        callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        self.callbacks = [callback_checkpoint, callback_val_loss]
        
    
        # CNN specific non-inherited properties
        self.input_type_im_or_chain = input_type #True: image, False: TChain
        self.input_shape = None
        self.input_layer = None
        
        
    
    def create_model(self, input_shape, kernel_widths=[3,3,3], n_filters=30, n_dense=100, exp_sig_bg_ratio=(154943/(7475938+8754330))) -> None:
        """
        Create a new model with given parameters.

        Args:
            input_shape (list): input shape of images
            kernel_widths (list, optional): Defaults to [3,3,3].
            n_filters (int, optional): Defaults to 30.
            n_dense (int, optional): Defaults to 100.
            exp_sig_bg_ratio (tuple, optional): n_signal/n_background to compute initial bias with output layer constant bias initializer. Defaults to (154943/(7475938+8754330)). https://www.tensorflow.org/tutorials/structured_data/imbalanced_datas
        """
        
        self.input_shape = input_shape
        inital_output_bias = tf.keras.initializers.Constant(np.log(exp_sig_bg_ratio)) #sig bg imbalance
        
        layers = []
        layers.append(tf.keras.layers.Input(shape=input_shape, name="input_layer"))
            
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(1,1,1), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_0")(layers[-1]))
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(2,2,2), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_1")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_0")(layers[-1]))
        layers.append(tf.keras.layers.MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1), name="max_pool_0")(layers[-1]))
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(2,2,1), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_2")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_1")(layers[-1]))
        layers.append(tf.keras.layers.MaxPooling3D(pool_size=(2,2,1), strides=(4,4,1), name="max_pool_1")(layers[-1])) #(4,4,1)
        #add more layers here
        layers.append(tf.keras.layers.Flatten(name="flatten")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_2")(layers[-1]))
        layers.append(tf.keras.layers.Dense(n_dense, activation='relu', name="dense_0")(layers[-1]))
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid', name="dense_1", bias_initializer=inital_output_bias)(layers[-1]))
        
        self.model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
        self.model.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)
        self.model.summary()
    
    
    def __create_detailed_model(self, input_shape, kernel_widths=[3,3,3], n_filters=30, n_dense=100, exp_sig_bg_ratio=(154943/(7475938+8754330))) -> None: #use for CV scans
        self.input_shape = input_shape
        inital_output_bias = tf.keras.initializers.Constant(np.log(exp_sig_bg_ratio)) #sig bg imbalance
        
        layers = []
        layers.append(tf.keras.layers.Input(shape=input_shape, name="input_layer"))
            
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(1,1,1), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_0")(layers[-1]))
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(2,2,2), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_1")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_0")(layers[-1]))
        layers.append(tf.keras.layers.MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1), name="max_pool_0")(layers[-1]))
        layers.append(tf.keras.layers.Conv3D(n_filters, kernel_widths, strides=(2,2,1), kernel_initializer=tf.keras.initializers.RandomUniform(), padding='same', name="conv3d_2")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_1")(layers[-1]))
        layers.append(tf.keras.layers.MaxPooling3D(pool_size=(4,4,1), strides=(4,4,1), name="max_pool_1")(layers[-1]))
        #add more layers here
        layers.append(tf.keras.layers.Flatten(name="flatten")(layers[-1]))
        layers.append(tf.keras.layers.BatchNormalization(name="batch_norm_2")(layers[-1]))
        layers.append(tf.keras.layers.Dense(n_dense, activation='relu', name="dense_0")(layers[-1]))
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid', name="dense_1", bias_initializer=inital_output_bias)(layers[-1]))
        
        self.model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
        self.model.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)
        self.model.summary()
    
    
    def load_model(self, config_file) -> None:
        """
        Load model config from file.

        Args:
            config_file (str): directory to model file.
        """
        
        self.model = tf.keras.models.load_model(config_file)
        self.model_name = config_file.split("/")[-1].split(".")[0]
        self.input_shape = self.model.input_shape[1:]
        self.model.summary()
        
    def save_model(self, model_name="") -> None:
        """
        Save model config to file.

        Args:
            model_name (str): name of the model file to be saved. Defaults to "[self.model_name].keras"
        """
        
        if model_name == "": model_name = self.model_name
        
        self.model.save(model_name+".keras")
        self.model_name = model_name
        
        
    def __update_data_splitting(self, input_data, split_ratio=[], batch_size=500, h5_blocksize=200000) -> None:
        if len(split_ratio) == 0: split_ratio = self.split_ratio
        
        n_signal_samples, n_background_samples = 0, 0
        data_path, tchains_data = "", []
        
        if self.input_type_im_or_chain:
            data_path = input_data
            if not os.path.exists(data_path):
                print("Path does not exist!")
                return
            
            n_background_files = len([name for name in os.listdir(data_path) if name.startswith("background_") and name.endswith(".hdf5")])
            n_signal_files = len([name for name in os.listdir(data_path) if name.startswith("signal") and name.endswith(".hdf5")])
            
            n_background_samples = (n_background_files-1) * h5_blocksize + len(h5py.File(data_path+"background_"+str(n_background_files-1)+".hdf5", 'r').keys())
            n_signal_samples = (n_signal_files-1) * h5_blocksize + len(h5py.File(data_path+"signal_"+str(n_signal_files-1)+".hdf5", 'r').keys())
        
        else:
            tchains_data = input_data
            if tchains_data is list: #expects list of TChains [signal, background]
                print("Wrong input type! data_path should be a list of TChain [signal, background]! type(input_data) =", type(input_data))
                return
            
            n_signal_samples = tchains_data[0].GetEntries()
            n_background_samples = tchains_data[1].GetEntries()
        
        # train/val/test splitting
        n_background_samples_train = int(n_background_samples * split_ratio[0])
        n_signal_samples_train = int(n_signal_samples * split_ratio[0])
        
        n_background_samples_val = int(n_background_samples * split_ratio[1])
        n_signal_samples_val = int(n_signal_samples * split_ratio[1])
        
        n_background_samples_test = n_background_samples - (n_background_samples_train + n_background_samples_val)
        n_signal_samples_test = n_signal_samples - (n_signal_samples_train + n_signal_samples_val)
        
        # train splitting/file management
        sig_bg_ratio = n_signal_samples_train / n_background_samples_train
        batch_size_signal = int(batch_size / (1+1/sig_bg_ratio))
        batch_size_background = batch_size - batch_size_signal
        
        n_batches = int(n_signal_samples_train / batch_size_signal)
        
        # validation splitting/file management
        sig_bg_ratio_val = n_signal_samples_val / n_background_samples_val
        batch_size_signal_val = int(batch_size / (1+1/sig_bg_ratio_val))
        batch_size_background_val = batch_size - batch_size_signal_val
        
        n_batches_val = int(n_signal_samples_val / batch_size_signal_val)
        
        val_offset_background = n_background_samples_train
        val_offset_signal = n_signal_samples_train
        
        # test splitting/file management
        sig_bg_ratio_test = n_signal_samples_test / n_background_samples_test
        batch_size_signal_test = int(batch_size / (1+1/sig_bg_ratio_test))
        batch_size_background_test = batch_size - batch_size_signal_test
        
        n_batches_test = int(n_signal_samples_test / batch_size_signal_test)
        
        test_offset_background = n_background_samples_train + n_background_samples_val
        test_offset_signal = n_signal_samples_train + n_signal_samples_val
            
        self.data_splitting = type('', (), {})() #does this work?
        self.data_splitting.data_path = data_path
        self.data_splitting.tchains_data = tchains_data
        
        self.data_splitting.h5_blocksize = h5_blocksize
        
        self.data_splitting.batch_size_signal = batch_size_signal
        self.data_splitting.batch_size_background = batch_size_background
        self.data_splitting.batch_size_signal_val = batch_size_signal_val
        self.data_splitting.batch_size_background_val = batch_size_background_val
        self.data_splitting.batch_size_signal_test = batch_size_signal_test
        self.data_splitting.batch_size_background_test = batch_size_background_test
        
        self.data_splitting.n_batches = n_batches
        self.data_splitting.n_batches_val = n_batches_val
        self.data_splitting.n_batches_test = n_batches_test
        
        self.data_splitting.train_offset_background = 0
        self.data_splitting.train_offset_signal = 0
        self.data_splitting.val_offset_background = val_offset_background
        self.data_splitting.val_offset_signal = val_offset_signal
        self.data_splitting.test_offset_background = test_offset_background
        self.data_splitting.test_offset_signal = test_offset_signal
        
        print("Data splitting updated!:")
        print("\tsplit_ratio =", split_ratio, "\tbatch_size =", batch_size, "\th5_blocksize =", h5_blocksize)
        print("\ttrain: signal =", n_signal_samples_train, "\tbackground =", n_background_samples_train, "\tbatch_size_signal =", batch_size_signal, "\tbatch_size_background =", batch_size_background, "\tn_batches =", n_batches)
        print("\tval: signal =", n_signal_samples_val, "\tbackground =", n_background_samples_val, "\tbatch_size_signal =", batch_size_signal_val, "\tbatch_size_background =", batch_size_background_val, "\tn_batches =", n_batches_val)
        print("\ttest: signal =", n_signal_samples_test, "\tbackground =", n_background_samples_test, "\tbatch_size_signal =", batch_size_signal_test, "\tbatch_size_background =", batch_size_background_test, "\tn_batches =", n_batches_test)
        print("\ttotal: signal =", n_signal_samples, "\tbackground =", n_background_samples, '\n')
        
        return
    
        
    def __load_data(self, step, type) -> Tuple[np.array, np.array]: #loads a single batch of data corresponding to the given step
        if type == "train":
            batch_size_signal = self.data_splitting.batch_size_signal
            batch_size_background = self.data_splitting.batch_size_background
            offset_signal = self.data_splitting.train_offset_signal
            offset_background = self.data_splitting.train_offset_background
        elif type == "val":
            batch_size_signal = self.data_splitting.batch_size_signal_val
            batch_size_background = self.data_splitting.batch_size_background_val
            offset_signal = self.data_splitting.val_offset_signal
            offset_background = self.data_splitting.val_offset_background
        elif type == "test":
            batch_size_signal = self.data_splitting.batch_size_signal_test
            batch_size_background = self.data_splitting.batch_size_background_test
            offset_signal = self.data_splitting.test_offset_signal
            offset_background = self.data_splitting.test_offset_background
        
        
        if self.input_type_im_or_chain: #hdf5
            data_path = self.data_splitting.data_path
            h5_blocksize = self.data_splitting.h5_blocksize
            
            
            continue_next_file_background = False
            current_file_background = int((step * batch_size_background + offset_background) / h5_blocksize)
            start_index_background = (step * batch_size_background + offset_background) % h5_blocksize
            end_index_background = start_index_background + batch_size_background
            
            if end_index_background > h5_blocksize:
                end_index_background = h5_blocksize
                continue_next_file_background = True
                
            f_background = h5py.File(data_path+"background_"+str(current_file_background)+".hdf5", 'r')
            keys_background = list(f_background.keys())[start_index_background:end_index_background]
            images_background = np.array([f_background[key] for key in keys_background])
            f_background.close()
            
            if continue_next_file_background:
                f_background = h5py.File(data_path+"background_"+str(current_file_background+1)+".hdf5", 'r')
                keys_background = list(f_background.keys())[0:end_index_background-h5_blocksize]
                images_background = np.concatenate((images_background, [f_background[key] for key in keys_background]), axis=0)
                f_background.close()
                
            continue_next_file_signal = False
            current_file_signal = int((step * batch_size_signal + offset_signal) / h5_blocksize)
            start_index_signal = (step * batch_size_signal + offset_signal) % h5_blocksize
            end_index_signal = start_index_signal + batch_size_signal
            
            if end_index_signal > h5_blocksize:
                end_index_signal = h5_blocksize
                continue_next_file_signal = True
                
            f_signal = h5py.File(data_path+"signal_"+str(current_file_signal)+".hdf5", 'r')
            keys_signal = list(f_signal.keys())[start_index_signal:end_index_signal]
            images_signal = np.array([f_signal[key] for key in keys_signal])
            f_signal.close()
            
            if continue_next_file_signal:
                f_signal = h5py.File(data_path+"signal_"+str(current_file_signal+1)+".hdf5", 'r')
                keys_signal = list(f_signal.keys())[0:end_index_signal-h5_blocksize]
                images_signal = np.concatenate((images_signal, [f_signal[key] for key in keys_signal]), axis=0)
                f_signal.close()
                
            images = np.concatenate((images_background, images_signal), axis=0)
            
        else: #TChain
            start_index_signal = step * batch_size_signal + offset_signal
            start_index_background = step * batch_size_background + offset_background
            
            end_index_signal = start_index_signal + batch_size_signal
            end_index_background = start_index_background + batch_size_background
            
            self.image_maker = CNNimageMaker(eta=[self.input_shape[0], -0.3, 0.3], phi=[self.input_shape[1], -np.pi/2, np.pi/2], use_root_histos=False, root_numpy_available=False)
            self.tree_holder_sig = TreeHolder(self.data_splitting.tchains_data[0])
            self.tree_holder_bg = TreeHolder(self.data_splitting.tchains_data[1])
            
            images = []
            
            if end_index_background > self.data_splitting.tchains_data[1].GetEntries():
                end_index_background = self.data_splitting.tchains_data[1].GetEntries()
            for entry in range(start_index_background, end_index_background):
                try:
                    self.data_splitting.tchains_data[1].GetEntry(entry)
                    self.tree_holder_bg.update()
                        
                    if self.verbosity > 0 and (entry-start_index_background) % 10**int(np.math.log10(batch_size_background)) == 0:
                        print("BG entry", str(entry-start_index_background) + '/' + str(batch_size_background), "("+str(round((entry-start_index_background)/batch_size_background*100, 2))+"%)", entry, start_index_background, end_index_background, end='\r')

                    for pj in range(self.tree_holder_bg.jet_width.size()):
                        jet_image = self.image_maker.MakeImage(self.tree_holder_bg, pj)
                        if (jet_image is int): #not (type(jet_image) is list)
                            print("ERROR: pj =", pj, "entry =", entry, "code =", jet_image)
                        else:
                            layers = ["bar", "end", "ext"]
                            for i in range(len(jet_image)):
                                image = jet_image[i]
                                image = image.squeeze()
                                jet_image[i] = image
                            big_image = np.concatenate((jet_image[0], jet_image[1], jet_image[2]), axis=2)
                            images.append(big_image)
                except:
                    print(entry, start_index_background, end_index_background)
            
            if end_index_signal > self.data_splitting.tchains_data[0].GetEntries():
                end_index_signal = self.data_splitting.tchains_data[0].GetEntries()
            for entry in range(start_index_signal, end_index_signal):
                try:
                    self.data_splitting.tchains_data[0].GetEntry(entry)
                    self.tree_holder_sig.update()
                        
                    if self.verbosity > 0 and (entry-start_index_signal) % 10**int(np.math.log10(batch_size_signal)) == 0:
                        print("SIG entry", str(entry-start_index_signal) + '/' + str(batch_size_signal), "("+str(round((entry-start_index_signal)/batch_size_signal*100, 2))+"%)", end='\r')

                    for pj in range(self.tree_holder_sig.jet_width.size()):
                        jet_image = self.image_maker.MakeImage(self.tree_holder_sig, pj)
                        if (jet_image is int): #not (type(jet_image) is list)
                            print("ERROR: pj =", pj, "entry =", entry, "code =", jet_image)
                        else:
                            layers = ["bar", "end", "ext"]
                            for i in range(len(jet_image)):
                                image = jet_image[i]
                                image = image.squeeze()
                                jet_image[i] = image
                            big_image = np.concatenate((jet_image[0], jet_image[1], jet_image[2]), axis=2)
                            images.append(big_image)
                except:
                    print(entry, start_index_signal, end_index_signal)
            
            images = np.array(images)
            
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], images.shape[3], 1)
        labels = np.concatenate((np.zeros(batch_size_background), np.ones(batch_size_signal)), axis=0)
            
        return (images, labels)
    
    
    """def test_loading(self, data, batch_size, h5_blocksize=200000) -> None:
        self.split_ratio = [0.6, 0.2, 0.2]
        self.__update_data_splitting(data, self.split_ratio, batch_size, h5_blocksize)
        n_batches_test = self.data_splitting.n_batches_test
        
        def test_generator(batch_size, n_batches):
            while True:
                for step in range(n_batches_test):
                    yield self.__load_data(step, "test")
        
        batched_test_data = tf.data.Dataset.from_generator(test_generator, args=[batch_size, self.data_splitting.n_batches_test], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        predictions = self.model.predict(batched_test_data)#, batch_size=batch_size, steps=self.data_splitting.n_batches_test)
    """ 
        
    
    def __train_model_hdf5(self, data_path, n_epochs, batch_size, split_ratio=[], use_gpu=False, h5_blocksize=200000) -> None:
        if batch_size > h5_blocksize:
            print("batch_size too large! Maximum is", h5_blocksize)
            return
        
        if len(split_ratio) == 0: split_ratio = self.split_ratio
        
        
        self.__update_data_splitting(data_path, split_ratio, batch_size, h5_blocksize)
        n_batches = self.data_splitting.n_batches
        n_batches_val = self.data_splitting.n_batches_val
        
        def train_generator(batch_size, n_batches):
            while True:
                for step in range(n_batches):
                    yield self.__load_data(step, "train")
                    
        def val_generator(batch_size, n_batches_val):
            while True:
                for step in range(n_batches_val):
                    yield self.__load_data(step, "val")
                    
        
        batched_train_data = tf.data.Dataset.from_generator(train_generator, args=[batch_size, n_batches], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        batched_val_data = tf.data.Dataset.from_generator(val_generator, args=[batch_size, n_batches_val], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.history = self.model.fit(batched_train_data, validation_data=batched_val_data, epochs=n_epochs, batch_size=batch_size, steps_per_epoch=n_batches, validation_steps=n_batches_val, callbacks=self.callbacks, shuffle=True) #could set steps=None to auto use entire dataset
        else: 
            self.history = self.model.fit(batched_train_data, validation_data=batched_val_data, epochs=n_epochs, batch_size=batch_size, steps_per_epoch=n_batches, validation_steps=n_batches_val, callbacks=self.callbacks, shuffle=True) #could set steps=None to auto use entire dataset
        
        return self.history
    
        
    def __train_model_root_data(self, data_tchains, n_epochs, batch_size, split_ratio=[], use_gpu=False) -> None:
        if self.input_type_im_or_chain:
            print("Wrong input type! Use create_model() for image input type.")
            return
        
        if len(split_ratio) == 0: split_ratio = self.split_ratio
        
        tchain_sig, tchain_bg = data_tchains[0], data_tchains[1]
        
        self.__update_data_splitting(data_tchains, split_ratio, batch_size)
        n_batches = self.data_splitting.n_batches
        n_batches_val = self.data_splitting.n_batches_val
        
        def train_generator(batch_size, n_batches):
            while True:
                for step in range(n_batches):
                    yield self.__load_data(step, "train")
                    
        def val_generator(batch_size, n_batches_val):
            while True:
                for step in range(n_batches_val):
                    yield self.__load_data(step, "val")
        
        batched_train_data = tf.data.Dataset.from_generator(train_generator, args=[batch_size, n_batches], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        batched_val_data = tf.data.Dataset.from_generator(val_generator, args=[batch_size, n_batches_val], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.history = self.model.fit(batched_train_data, validation_data=batched_val_data, epochs=n_epochs, batch_size=batch_size, steps_per_epoch=n_batches, validation_steps=n_batches_val, callbacks=self.callbacks, shuffle=True) #could set steps=None to auto use entire dataset
        else:
            self.history = self.model.fit(batched_train_data, validation_data=batched_val_data, epochs=n_epochs, batch_size=batch_size, steps_per_epoch=n_batches, validation_steps=n_batches_val, callbacks=self.callbacks, shuffle=True) #could set steps=None to auto use entire dataset
            
        return self.history
        
        
    def train_model(self, data, n_epochs, batch_size, split_ratio=[], use_gpu=False, h5_blocksize = 200000, continue_training=False) -> None:
        """
        Train model on given data (path to hdf5 file directory or list of TChains [signal, background]).

        Args:
            data (str/list): path to hdf5 file directory or list of TChains [signal, background] (must match input_type_im_or_chain property!)
            n_epochs (int): number of epochs
            batch_size (int): batch size (must be < h5_blocksize)
            split_ratio (list, optional): split ratio of data ([train/val/test]). Defaults to split_ratio default split ratio of class.
            use_gpu (bool, optional): run training on '/device:GPU:0'. Defaults to False.
            h5_blocksize (int, optional): size of dataset blocks in HDF5 input files (must match batch size of input files!). Defaults to 200000.
            continue_training (bool, optional): continue training from last checkpoint (if exists). Defaults to False.
        """
        
        #check if model directory exists
        if not os.path.exists("models/" + self.model_name):
            print("creating", "models/" + self.model_name)
            os.makedirs("models/" + self.model_name)
        
        #continue training from last checkpoint
        if continue_training:
            latest_checkpoint = 0
            for file in os.listdir("models/" + self.model_name):
                if file.endswith(self.checkpoint_format):
                    checkpoint = int(file.split("-")[1].split(".")[0])
                    if checkpoint > latest_checkpoint:
                        latest_checkpoint = checkpoint
            if latest_checkpoint > 0:                
                self.model.load("models/" + self.model_name + "/cp-" + str(latest_checkpoint).zfill(4) + self.checkpoint_format) #load_weights? self.load()?
                n_epochs = n_epochs - latest_checkpoint
                print("Loaded model from checkpoint", latest_checkpoint)
                print("Remaining epochs:", n_epochs)
            else:
                print("No checkpoint found! Starting from scratch!")
        
        if self.model is None:
            print("No model loaded!")
            return
        
        if self.input_type_im_or_chain:
            if type(data) != str:
                print("Wrong input type! Expected: string (path to hdf5 file directory)")
                return
            self.__train_model_hdf5(data, n_epochs, batch_size, split_ratio, use_gpu, h5_blocksize)
        else:
            if data is list:
                print("Wrong input type! Expected: list of TChains [signal, background]")
                return
            self.__train_model_root_data(data, n_epochs, batch_size, split_ratio, use_gpu)
    
    
    def train_scan(self, data:str, batch_size, input_shape, param_file, n_epochs=10, fold=0, n_fold=1, split_ratio=[], use_gpu=False, h5_blocksize = 200000) -> None:
        """
        Runs a single scan with a parameter config specified in a file.

        Args:
            data (str): HDF5 file directory (no TChains - this makes no sense for a parallelized scan)
            batch_size (int): batch size (must be < h5_blocksize)
            input_shape (list): input shape of images (must match image shapes in input files)
            param_file (str): path to parameter file
            n_epochs (int, optional): number of epochs. Defaults to 10.
            fold (int, optional): # of fold for n-fold scan. Defaults to 0.
            n_fold (int, optional): # of total folds for n-fold scan. Defaults to 1.
            split_ratio (list, optional): split ratio of data ([train/val/test]). Defaults to split_ratio default split ratio of class.
            use_gpu (bool, optional): run training on '/device:GPU:0'. Defaults to False.
            h5_blocksize (int, optional): size of dataset blocks in HDF5 input files (must match batch size of input files!). Defaults to 200000.
        """
        
        try: #read parameter dict from file
            import pickle
            param_no = int(param_file.split("/")[-1].split(".")[0])
            model_out_dir = param_file.split("/")[-2] + "/models/"
            results_out_dir = param_file.split("/")[-2] + "/results/"
            with open(param_file, 'rb') as file_pi:
                params = pickle.load(file_pi)
            #param_file_in = open(param_file, "r")
            #params = eval(param_file_in.read())
            #param_file_in.close()
        except:
            print("Could not read param file!")
            return
            
        self.__create_detailed_model(input_shape, params)
        self.train_model(data, n_epochs, batch_size, split_ratio, use_gpu, h5_blocksize)
        self.save_model(model_out_dir + self.model_name + '_' + str(param_no) + '_' + str(fold))
        
        with open(results_out_dir + self.model_name + '_' + str(param_no) + '_' + str(fold) + ".history", 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)
        return
    
    
    def predict(self):
        pass
        