import sys, os
sys.path.append(os.path.dirname(__file__))

from NNjetTagger import NNjetTagger

import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torchmetrics.functional import accuracy

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.nn import GCNConv, GATv2Conv, ChebConv, ARMAConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

import pytorch_lightning as ptlight
from pytorch_lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers


from sklearn import metrics
import numpy as np

from abc import abstractmethod
import warnings


class GNNjetTagger_modelWrapper(LightningModule, NNjetTagger): # model needs separate class, because it can't reference itself to trainer.fit() but needs to inherit from NNjetTagger for metrics functionalities, etc
    def __init__(self, hidden_channels, node_feat_size=None, use_edge_attr=False, learning_rate=0.001, loss_func=None):
        super(GNNjetTagger_modelWrapper, self).__init__() #(nn_type=False, model_name="GNNjetTagger_modelWrapper")
        
        self.hidden_channels = hidden_channels
        self.node_features_size = node_feat_size if node_feat_size else 4
        self.use_edge_attr = use_edge_attr

        self.loss = loss_func if loss_func else torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        
        
        self.norm = BatchNorm(self.node_features_size)
        self.conv1 = ARMAConv(self.node_features_size, self.hidden_channels, num_stacks=3)
        self.conv2 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks=3)
        self.conv3 = ARMAConv(self.hidden_channels, self.hidden_channels, num_stacks=3)
        self.lin0 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin = Linear(self.hidden_channels, 1)
        
        self.save_hyperparameters()
    
    #@abstractmethod
    def forward(self, mini_batch):#x, edge_index, mini_batch): # switch for GNNExplainer
        # 0. Unbatch elements in mini batch
        x, edge_index, batch = mini_batch.x, mini_batch.edge_index, mini_batch.batch
        edge_attr = mini_batch.edge_attr if self.use_edge_attr else None

        # 1. Apply Batch normalization
        x = self.norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)

        # 3. Readout layer
        y1 = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        y2 = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        y3 = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        z = self.lin0(y1 + y2 + y3).relu()
        
        # 4. Apply a final classifier
        z = F.dropout(z, p=0.5, training=self.training)
        z = self.lin(z)
        
        return z

    def training_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        labels = batch.y.unsqueeze(1).float()
        
        loss = self.loss(out, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        metric_scores = {}
        for metric in self.metrics:
            metric_scores[metric.__name__] = metric(labels.detach().cpu().numpy(), np.round(predictions))
            self.log('train_' + metric.__name__, metric_scores[metric.__name__], on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        labels = batch.y.unsqueeze(1).float()

        loss = self.loss(out, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        metric_scores = {}
        for metric in self.metrics:
            metric_scores[metric.__name__] = metric(labels.detach().cpu().numpy(), np.round(predictions))
            self.log('val_' + metric.__name__, metric_scores[metric.__name__], on_step=False, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx):
        out = self(batch)  # Perform a single forward pass.
        predictions = torch.sigmoid(out).detach().cpu().numpy()
        #labels = batch.y.unsqueeze(1).float()
        return predictions.tolist()

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min = 0.00001)
        return [self.optimizer], [scheduler]
        


class GNNjetTagger(NNjetTagger):
    def __init__(self, model_name="GNNjetTagger", seed=69420):
        super(GNNjetTagger, self).__init__(False, model_name)
        
        torch.manual_seed(seed)
        
        # overwrite superclass defaults with GNN specific defaults
        self.metrics = [metrics.accuracy_score, metrics.precision_score, metrics.recall_score, metrics.f1_score] #["accuracy", "precision", "recall", "f1"]
        self.metrics_labels = ["Accuracy", "Precision", "Recall", "F1"]
        
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        self.learning_rate = 0.001
        
        self.checkpoint_format = ".ckpt"
        self.checkpoint_path = "models/" + model_name + "/cp-{epoch:04d}" + self.checkpoint_format
        
        callback_checkpoint = ptlight.callbacks.ModelCheckpoint(dirpath=self.checkpoint_path, filename='gcn-{epoch:02d}', every_n_epochs=1, save_top_k=-1)
        callback_val_loss = ptlight.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=7, verbose=True, mode="min")
        callback_lr_monitor = ptlight.callbacks.LearningRateMonitor(logging_interval='step')
        self.callbacks = [callback_checkpoint, callback_val_loss, callback_lr_monitor]
        
        # GNN specific non-inherited properties
        self.node_features_size = None
        self.hidden_channels = 256
        
        self.check_val_every_n_epoch = 1
        
        
    def create_model(self, node_features_size:int, hidden_channels=256):
        self.node_features_size = node_features_size
        self.hidden_channels = hidden_channels
        
        self.model = GNNjetTagger_modelWrapper(hidden_channels=self.hidden_channels, 
                                               node_feat_size=self.node_features_size, 
                                               learning_rate=self.learning_rate, 
                                               loss_func=self.loss)
        
        self.model.metrics = self.metrics
    
    
    def load_model(self, config_file) -> None:
        """
        Load model config from file.

        Args:
            config_file (str): directory to model file.
        """
        if self.jetgraph_model is None:
            print("Need to define a model architecture first.")
        
        self.model = self.load_from_checkpoint(config_file)
        self.model_name = config_file.split("/")[-1].split(".")[0]
        self.model.summary()
        
    
    def __update_data_splitting(self):
        pass
     
        
    def __load_data(self):
        pass
    
        
    def train_model(self, train_loader, val_loader, n_epochs=10):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        trainer = ptlight.Trainer(
            #devices=1 if str(device).startswith("cuda") else 0,
            #accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            default_root_dir='./', 
            max_epochs=n_epochs, 
            callbacks=self.callbacks,
            enable_progress_bar=True,
            log_every_n_steps=5,
            check_val_every_n_epoch=self.check_val_every_n_epoch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(self.model, train_loader, val_loader)