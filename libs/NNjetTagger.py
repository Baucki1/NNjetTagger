
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#super class for CNNjetTagger and GNNjetTagger

class NNjetTagger:
    def __init__(self, nn_type:bool, model_name:str):
        #super(NNjetTagger, self).__init__(False, model_name)
        self.verbosity = 0
        
        self.nn_type = nn_type
        
        self.model = None
        self.model_name = model_name
        
        #training histories
        self.history = None
        self.train_val_history = None
        self.test_history = None
        
        #data preparation
        self.split_ratio = [0.6, 0.2, 0.2]
        
        #classification threshold
        self.threshold = 0.75
        
        #metrics
        self.metrics = []
        self.metrics_labels = []
        
        self.loss = None
        
        #training parameters
        self.optimizer = None
        self.learning_rate = 0.001
        
        
        #training checkpoints
        self.checkpoint_format = ".ckpt"
        self.checkpoint_path = "models/" + model_name + "/cp-{epoch:04d}" + self.checkpoint_format
        
        #callbacks
        self.callbacks = []
    
    
    @abstractmethod
    def create_model(self):
        pass
    
    
    @abstractmethod
    def load_model(self, model_name="") -> None:
        pass
    
    
    def save_model(self, model_name="") -> None:
        """
        Save model config to file.

        Args:
            model_name (str): name of the model file to be saved. Defaults to self.model_name
        """
        
        if model_name == "": model_name = self.model_name
        
        self.model.save(model_name)
        self.model_name = model_name
        
        return
    
    
    @abstractmethod
    def train_model(self):
        pass
        
        
    @abstractmethod
    def predict(self):
        pass
    
    
    def measure_performance(self, data, batch_size, split_ratio=[], h5_blocksize=200000, use_gpu=False, plot_train_val_history=False, plot_train_performance=False, plot_val_performance=False, plot_test_performance=True, plot_performance_evt_quantities=[], log_scale=[], scale=[], names=[]) -> None:
        """
        Standard routine to measure performance of the model and plot history of metrics, compute scores, etc.

        Args:
            data (str or list): path to input data used to measure training performance or list of TChains [signal, background] (must match input_type_im_or_chain property!)
            batch_size (int): batch size for loading data (must be < h5_blocksize)
            split_ratio (list, optional): split ratio of data ([train/val/test]). Defaults to split_ratio default split ratio of class.
            h5_blocksize (int, optional): size of dataset blocks in HDF5 input files (must match batch size of input files!). Defaults to 200000.
            use_gpu (bool, optional): run training on '/device:GPU:0'. Defaults to False.
            plot_train_val_history (bool, optional): plot training history of metrics for train/val data (might not be available if model was pre-trained/loaded from file). Defaults to False.
            plot_train_performance (bool, optional): measure performance on train data and add to history plot. Defaults to True.
            plot_val_performance (bool, optional): measure performance on val data and add to history plot. Defaults to True.
            plot_test_performance (bool, optional): measure performance on test data and add to history plot. Defaults to True.
            plot_performance_evt_quantities (list, optional): plot predictions vs event quantities (only available for TChain input type). Defaults to [].
            log_scale (list, optional): [True|False] use log scale for quantity axes. Defaults to True.
        """
        
        if self.model is None:
            print("No model loaded!")
            return
        
        if plot_performance_evt_quantities != [] and self.input_type_im_or_chain == True:
            print("Event quantities only available for TChain input type! (input_type_im_or_chain = True and plot_performance_evt_quantities not empty)")
            return
        
        if plot_performance_evt_quantities != [] and data is list:
            print("Event quantities only available for TChain input type! (Expected: list of TChains [signal, background] as data_path)")
            return
        
        if len(split_ratio) == 0: split_ratio = self.split_ratio
        
        self.__update_data_splitting(data, split_ratio, batch_size, h5_blocksize)
        n_batches_test = self.data_splitting.n_batches_test
        
        
        def test_generator(batch_size, n_batches):
            while True:
                for step in range(n_batches_test):
                    yield self.__load_data(step, "test")
                    
        def histedges_equalN(x, nbin):
            npt = len(x)
            return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))
                
        def plot_metric_history(metric_name, metric_label="", test_history=None, train_val_history=None):
            if (plot_train_val_history or plot_test_history) == False: return
            
            try:
                if plot_train_val_history and not (train_val_history == None): plt.plot(train_val_history.history[metric_name], label='train')
                if plot_train_val_history and not (train_val_history == None): plt.plot(train_val_history.history['val_'+metric_name], label='val')
                if plot_test_history and not (test_history == None): plt.plot(test_history['test_'+metric_name], label='test')
            except KeyError:
                print("Metric not found in history!")
                return
            if metric_label == "": metric_label = metric_name
            plt.title(self.model_name + ' ' + metric_label)
            plt.ylabel(metric_name)
            plt.xlabel('epoch')
            plt.legend(loc='upper left')
            plt.show()
        
        def plot_quantity_vs_prediction(quantity, prediction, labels, branch_name):
            plt.scatter(quantity, prediction, c=labels, cmap='coolwarm', s=0.5)
            #plt.colorbar()
            if log_scale[plot_performance_evt_quantities.index(branch_name)]: plt.xscale('log')
            #plt.locator_params(axis='x', nbins=7)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("Prediction")
            plt.title("Prediction vs " + branch_name)
            #plt.colorbar(ticks=[0, 1], format=mticker.FixedFormatter(["background", "signal"]))
            plt.show()
            
            quantity, prediction = quantity.flatten(), prediction.flatten()
            #2D histogram of prediction vs quantity distribution
            plt.hist2d(quantity, prediction, bins=(50, 50), cmap=plt.cm.viridis)
            plt.colorbar()
            if log_scale[plot_performance_evt_quantities.index(branch_name)]: plt.xscale('log')
            #plt.locator_params(axis='x', nbins=7)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("Prediction")
            plt.title("Prediction vs " + branch_name)
            plt.show()
            
            #2D histogram of prediction vs quantity distribution (signal only)
            plt.hist2d(quantity[labels==1], prediction[labels==1], bins=(50, 50), cmap=plt.cm.viridis)
            plt.colorbar()
            if log_scale[plot_performance_evt_quantities.index(branch_name)]: plt.xscale('log')
            #plt.locator_params(axis='x', nbins=7)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("Prediction")
            plt.title("Prediction vs " + branch_name + " (signal)")
            plt.show()
            
            #2D histogram of prediction vs quantity distribution (background only)
            plt.hist2d(quantity[labels==0], prediction[labels==0], bins=(50, 50), cmap=plt.cm.viridis)
            plt.colorbar()
            if log_scale[plot_performance_evt_quantities.index(branch_name)]: plt.xscale('log')
            #plt.locator_params(axis='x', nbins=7)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("Prediction")
            plt.title("Prediction vs " + branch_name + " (background)")
            plt.show()
        
        def plot_quantity_hist(quantity, labels, branch_name):
            quantity_sig, quantity_bg = quantity[labels==1], quantity[labels==0]
            plt.hist(quantity_sig, bins=50, alpha=0.5, label='signal', color='r')
            plt.hist(quantity_bg, bins=50, alpha=0.5, label='background', color='b')
            ks_score = ss.ks_2samp(quantity_sig, quantity_bg)
            #if log_scale[plot_performance_evt_quantities.index(branch_name)]: plt.xscale('log')
            plt.locator_params(axis='x', nbins=7)
            plt.legend(loc='upper right', title='KS score: '+str(round(ks_score[0], 4)))
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.title(branch_name + " distribution")
            plt.show()
        
        def plot_prediction_hist(prediction, labels):
            prediction_sig, prediction_bg = prediction[labels==1], prediction[labels==0]
            plt.hist(prediction_sig, bins=50, alpha=0.5, label='signal', color='r')
            plt.hist(prediction_bg, bins=50, alpha=0.5, label='background', color='b')
            plt.xlabel("Prediction")
            plt.title("Prediction distribution")
            plt.legend(loc='upper right')
            plt.show()
            
        def plot_metric_vs_quantity(metric, predictions, labels, quantity, metric_name, branch_name, nbins=15): # bin metric score in bins of quantity and plot mean metric score in each bin
            quantity = quantity * scale[plot_performance_evt_quantities.index(branch_name)] #rescale quantity
            bins = histedges_equalN(quantity, nbins) #bins = np.linspace(min(quantity), max(quantity), nbins)
            bin_indices = np.digitize(quantity, bins)
            metric_score_bin, metric_error_bin = [], []
            for i in range(len(bins)):
                prediction_bin = predictions[bin_indices == i]
                labels_bin = labels[bin_indices == i]
                metric_score_bin.append(metric(labels_bin, prediction_bin))
                #compute error per bin using bootstrapping
                metric_scores_boot, n_samples = [], 50
                for _ in range(n_samples):
                    indices = np.random.choice(len(prediction_bin), len(prediction_bin), replace=True)
                    metric_scores_boot.append(metric(labels_bin[indices], prediction_bin[indices]))
                metric_error_bin.append(np.std(metric_scores_boot))
            
                
                
            plt.errorbar(bins, metric_score_bin, yerr=metric_error_bin, fmt='x', color='red', ecolor='black', capsize=3, markersize=3)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel(metric_name)
            plt.title(metric_name + " vs " + branch_name)
            plt.show()
            
        def plot_tagging_eff_fake_rate_vs_quantity(prediction, labels, quantity, branch_name, threshold=self.threshold, nbins=15):
            quantity = quantity * scale[plot_performance_evt_quantities.index(branch_name)] #rescale quantity
            prediction = prediction.flatten()
            bins = histedges_equalN(quantity, nbins) #bins = np.linspace(min(quantity), max(quantity), nbins)
            bin_indices = np.digitize(quantity, bins)
            tagging_eff, fake_rate, tagging_eff_err, fake_rate_err = [], [], [], []
            tagging_eff_num, tagging_eff_denom, fake_rate_num, fake_rate_denom = [], [], [], []
            for i in range(len(bins)):
                prediction_bin = prediction[bin_indices==i]
                labels_bin = labels[bin_indices==i]
                labels_bin_sig_true, labels_bin_bg_true, prediction_bin_sig_true, prediction_bin_bg_true = labels_bin[labels_bin>=threshold], labels_bin[labels_bin<threshold], prediction_bin[labels_bin>=threshold], prediction_bin[labels_bin<threshold]
                labels_bin_sig, labels_bin_bg = labels_bin_sig_true[prediction_bin_sig_true >= threshold], labels_bin_bg_true[prediction_bin_bg_true >= threshold]
                n_bin_sig, n_bin_bg, n_bin_sig_true, n_bin_bg_true, n_bin = len(labels_bin_sig), len(labels_bin_bg), len(labels_bin_sig_true), len(labels_bin_bg_true), len(labels_bin)
                if n_bin_sig_true == 0: n_bin_sig_true = 1
                if n_bin_bg_true == 0: n_bin_bg_true = 1
                tagging_eff_num.append(n_bin_sig)
                tagging_eff_denom.append(n_bin_sig_true)
                fake_rate_num.append(n_bin_bg)
                fake_rate_denom.append(n_bin_bg_true)
                if len(prediction_bin) == 0: 
                    tagging_eff.append(0)
                    fake_rate.append(0)
                    tagging_eff_err.append(0)
                    fake_rate_err.append(0)
                    continue
                tagging_eff.append(n_bin_sig / n_bin_sig_true)
                fake_rate.append(n_bin_bg / n_bin_bg_true)
                tagging_eff_err.append(np.sqrt((np.sqrt(n_bin_sig))**2 + (np.sqrt(n_bin_sig_true)*n_bin_sig/n_bin_sig_true)**2)/n_bin_sig_true)
                fake_rate_err.append(np.sqrt((np.sqrt(n_bin_bg))**2 + (np.sqrt(n_bin_bg_true)*n_bin_bg/n_bin_bg_true)**2)/n_bin_bg_true)
            
            tagging_eff_num, tagging_eff_denom = np.array(tagging_eff_num), np.array(tagging_eff_denom)
            fake_rate_num, fake_rate_denom = np.array(fake_rate_num), np.array(fake_rate_denom)
               
            print(tagging_eff_num)
            print(tagging_eff_denom)
            print(tagging_eff_num/tagging_eff_denom)
            plt.errorbar(bins, tagging_eff, yerr=tagging_eff_err, label='tagging efficiency', fmt='x', color='red', ecolor='black', capsize=3, markersize=3)
            plt.ylim(0, 1)
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("tagging efficiency")
            plt.title("Tagging efficiency vs " + branch_name)
            plt.legend(loc='upper right')
            plt.show()
            
            plt.errorbar(bins, tagging_eff_num, color='red', label='numerator', fmt='x')
            plt.errorbar(bins,tagging_eff_denom, color='blue', label='denominator', fmt='x')
            plt.legend()
            plt.show()
            
            print(fake_rate_num)
            print(fake_rate_denom)
            print(fake_rate_num/fake_rate_denom)
            plt.errorbar(bins, fake_rate, yerr=fake_rate_err, label='fake rate', fmt='x', color='red', ecolor='black', capsize=3, markersize=3)
            plt.ylim(0, 1)
            plt.legend(loc='upper right')
            plt.xlabel(names[plot_performance_evt_quantities.index(branch_name)])
            plt.ylabel("fake rate")
            plt.title("Fake rate vs " + branch_name)
            plt.show()
            
            plt.errorbar(bins,fake_rate_num, color='red', label='numerator', fmt='x')
            plt.errorbar(bins,fake_rate_denom, color='blue', label='denominator', fmt='x')
            plt.legend()
            plt.show()
        
        batched_test_data = tf.data.Dataset.from_generator(test_generator, args=[batch_size, self.data_splitting.n_batches_test], output_types=(tf.float32, tf.float32), output_shapes=(np.insert(self.input_shape, 0, batch_size), [batch_size]))
        
        # plot predictions vs event quantities
        predictions, labels = [], np.array([])
        branches, quantities = [], []
        if plot_performance_evt_quantities != []:
            if use_gpu:
                    with tf.device('/device:GPU:0'):
                        predictions = self.model.predict(batched_test_data, batch_size=batch_size, steps=self.data_splitting.n_batches_test)
            else:
                predictions = self.model.predict(batched_test_data, batch_size=batch_size, steps=self.data_splitting.n_batches_test)
                
            
            for step in range(self.data_splitting.n_batches_test):
                labels_batch = np.concatenate((np.zeros(self.data_splitting.batch_size_background_test), np.ones(self.data_splitting.batch_size_signal_test)), axis=0)
                labels = np.concatenate((labels, labels_batch), axis=0)
            
            branches = [branch.GetName() for branch in data[0].GetListOfBranches()]
                
            for parameter in plot_performance_evt_quantities:
                if parameter in branches:
                    quantity = np.array([])
                    for step in range(self.data_splitting.n_batches_test):
                        if step * self.data_splitting.batch_size_background_test > data[1].GetEntries():
                            print("Current batch exceeds input TChain length (background). Resetting batch size...")
                            self.data_splitting.batch_size_background_test = data[1].GetEntries() - (step-1) * self.data_splitting.batch_size_background_test
                        for i in range(self.data_splitting.batch_size_background_test):
                            pos = step * self.data_splitting.batch_size_background_test + i
                            if pos >= data[1].GetEntries(): break 
                            data[1].GetEntry(pos)
                            branch_type, branch_entry = data[1].GetLeaf(parameter).GetTypeName(), 0
                            if "vector" in branch_type:
                                branch_entry = getattr(data[1], parameter).at(0)
                            else:
                                branch_entry = getattr(data[1], parameter)
                            branch_entry = np.array([branch_entry])
                            quantity = np.concatenate((quantity, branch_entry))
                            
                        if step * self.data_splitting.batch_size_signal_test > data[0].GetEntries():
                            print("Current batch exceeds input TChain length (signal). Resetting batch size...")
                            self.data_splitting.batch_size_signal_test = data[0].GetEntries() - (step-1) * self.data_splitting.batch_size_signal_test
                        for i in range(self.data_splitting.batch_size_signal_test):
                            pos = step * self.data_splitting.batch_size_signal_test + i
                            if pos >= data[0].GetEntries(): break
                            data[0].GetEntry(pos)
                            branch_type, branch_entry = data[0].GetLeaf(parameter).GetTypeName(), 0
                            if "vector" in branch_type:
                                branch_entry = getattr(data[0], parameter).at(0)
                            else:
                                branch_entry = getattr(data[0], parameter)
                            branch_entry = np.array([branch_entry])
                            quantity = np.concatenate((quantity, branch_entry))
                            
                    quantities.append(quantity)
                    plot_quantity_hist(quantity, labels, parameter)
                    plot_prediction_hist(predictions, labels)
                    plot_quantity_vs_prediction(quantity, predictions, labels, parameter)
                    plot_tagging_eff_fake_rate_vs_quantity(predictions, labels, quantity, parameter)
                    
                    for metric in self.metrics:
                        plot_metric_vs_quantity(metric, predictions, labels, quantity, self.metrics_labels[self.metrics.index(metric)], parameter)
                else:
                    print(parameter, "not found in TChain branches! (This might cause crashes)")
                    continue
        
        
        # measure performance on test data
        if plot_test_history:
            if predictions == [] and labels == []:
                if use_gpu:
                    with tf.device('/device:GPU:0'):
                        self.test_history = self.model.evaluate(batched_test_data)#, batch_size=batch_size, steps=self.data_splitting.n_batches_test)
                else:
                    self.test_history = self.model.evaluate(batched_test_data)#, batch_size=batch_size, steps=self.data_splitting.n_batches_test)
            else:
                self.test_history = type('', (), {})()
                for metric in self.metrics:
                    if metric.name == "binary_accuracy":
                        self.test_history.binary_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_"+metric.name)(labels, predictions)
                    elif metric.name == "auc":
                        self.test_history.auc = tf.keras.metrics.AUC(name="test_"+metric.name)(labels, predictions)
                    elif metric.name == "precision":
                        self.test_history.precision = tf.keras.metrics.Precision(name="test_"+metric.name)(labels, predictions)
                    elif metric.name == "true_positives":
                        self.test_history.true_positives = tf.keras.metrics.TruePositives(name="test_"+metric.name, thresholds=0.5)(labels, predictions)
                    elif metric.name == "true_negatives":
                        self.test_history.true_negatives = tf.keras.metrics.TrueNegatives(name="test_"+metric.name, thresholds=0.5)(labels, predictions)
                    elif metric.name == "false_positives":
                        self.test_history.false_positives = tf.keras.metrics.FalsePositives(name="test_"+metric.name, thresholds=0.5)(labels, predictions)
                    elif metric.name == "false_negatives":
                        self.test_history.false_negatives = tf.keras.metrics.FalseNegatives(name="test_"+metric.name, thresholds=0.5)(labels, predictions)
                    else:
                        print("Metric", metric.name, "not found!")
                        continue
                
                #this cannot work, metrics are single values computed for the entire dataset, need to bin the quantities and compute metric on every bin
                #for metric in self.test_history.__dict__.keys():
                #    for parameter in plot_performance_evt_quantities:
                #        quantity = quantities[plot_performance_evt_quantities.index(parameter)]
                #        print(self.test_history.binary_accuracy.shape, quantity.shape, labels.shape, predictions.shape)
                #        plot_metric_vs_quantity(self.test_history.__dict__[metric], quantity, labels, metric, parameter)
        
            
        
        # add in performance histories from training
        if plot_train_val_history: self.train_val_history = self.history
            
        
        #plot metric histories
        plot_metric_history('loss', metric_label="Loss ("+self.loss.name+')', test_history=self.test_history, train_val_history=self.train_val_history)
        for metric in self.metrics:
            plot_metric_history(metric.name, metric_label=self.metrics_labels[self.metrics.index(metric)], test_history=self.test_history, train_val_history=self.train_val_history)
        
        return
    
        
        