
import numpy as np
import h5py
import deepdish as dd

import ROOT
from ROOT import TFile

from NNimageMaker import CNNimageMaker, GNNgraphMaker
from TreeHolder import TreeHolder

class CNNdatasetConverter():
    def __init__(self, use_root_histos=False, root_numpy_available=False) -> None:
        self.image_maker = CNNimageMaker(use_root_histos=False, root_numpy_available=False)
        self.tree_reader = TreeHolder()
        return
    
    def convert_dataset(self, filename:str, in_dir:str, out_dir:str, h5_blocksize=200000, start_index=0, offset=0, filename_offset=0) -> None:
        """
        Convert a dataset from a root file to a hdf5 file.

        Args:
            filename (str): file name of input root file and also basename of output HDF5 files
            in_dir (str): directory of input root file
            out_dir (str): output directory to write converted HDF5 files to
            h5_blocksize (int, optional): # of datasets per HDF5 file. Defaults to 200000.
            start_index (int, optional): start index within tree in file. Defaults to 0.
            offset (int, optional): index offset for output (e.g. when converting multiple input rootfiles to same continuous collection of HDF5 files). Defaults to 0.
            filename_offset (int, optional): additional offset of HDF5 filename index. Defaults to 0.
        """
        
        savename = ''.join([i for i in filename if not i.isdigit()])

        file = TFile.Open(in_dir + filename + ".root")

        tree = file.Get("T")
        self.tree_reader.set_tree(tree)
        nentries = tree.GetEntries()

        
        i = filename_offset + int((offset+start_index)/h5_blocksize)
        hf = h5py.File(out_dir + savename + "_" + str(i) + ".hdf5", 'a')

        first_entry = True
        try:
            for entry in range(start_index, nentries):
                i = filename_offset + int((offset+entry)/h5_blocksize)
                
                tree.GetEntry(entry)
                self.tree_reader.update()

                if entry % h5_blocksize == 0:
                    hf = h5py.File(out_dir + savename + "_" + str(i) + ".hdf5", 'a')
                    
                if entry%1000 == 0:
                    print("entry", str(entry) + '/' + str(nentries), "("+str(round(entry/nentries*100, 2))+"%)", end='\r')

                for pj in range(self.tree_reader.jet_width.size()):
                    jet_image = self.image_maker.MakeImage(self.tree_reader, pj)
                    if not (type(jet_image) is list):
                        print("ERROR: pj =", pj, "entry =", entry, "code =", jet_image)
                    else:
                        layers = ["bar", "end", "ext"]
                        for i in range(len(jet_image)):
                            image = jet_image[i]
                            image = image.squeeze()
                            jet_image[i] = image
                        big_image = np.concatenate((jet_image[0], jet_image[1], jet_image[2]), axis=2)
                        hf.create_dataset(str(offset+entry)+"_"+str(pj), data=big_image, compression="gzip", compression_opts=4)
                if entry % h5_blocksize == h5_blocksize-1:
                    hf.close()
                if first_entry: first_entry = False
            hf.close()
        except KeyboardInterrupt:
            hf.close()
            print("Stopped at entry", str(entry) + '/' + str(nentries), "("+str(round(entry/nentries*100, 2))+"%)")
        return
    
    
    
class GNNdatasetConverter():
    def __init__(self) -> None:
        self.graph_maker = GNNgraphMaker()
        self.tree_reader = TreeHolder()
        
        self.n_layers = 4
        self.labels = ["barnodes", "barpos3", "barpos4", "bareta", "barphi", "barlayer", "barenergy", "barenergyabs"]
        return
    
    def convert_dataset(self, filename:str, in_dir:str, out_dir:str, h5_blocksize=200000, start_index=0, n_entries=0) -> None:
        """
        Convert a dataset from a root file to a hdf5 file.

        Args:
            filename (str): file name of input root file and also basename of output HDF5 files
            in_dir (str): directory of input root file
            out_dir (str): output directory to write converted HDF5 files to
            h5_blocksize (int, optional):
            start_index (int, optional): start index within tree in file. Defaults to 0.
            n_entries (int, optional): number of entries to convert. Defaults to 0.
        """

        file = TFile.Open(in_dir + filename)

        tree = file.Get("T")
        self.tree_reader.set_tree(tree)

        jet_graphs = []
        
        nentries = tree.GetEntries() if n_entries == 0 else n_entries
        
        if nentries > tree.GetEntries(): 
            print("WARNING: n_entries exceeds number of entries in tree. Using all entries.")
            nentries = tree.GetEntries()
        
        for entry in range(start_index, nentries):
            tree.GetEntry(entry)
            self.tree_reader.update()

            if entry%(int((nentries-start_index)/100)) == 0:
                print("entry", str(entry) + '/' + str(nentries), "("+str(round((entry-start_index)/(nentries-start_index)*100, 2))+"%)", end='\r')

            #print("n_jets=",tree_reader.jet_width.size())
            for pj in range(self.tree_reader.jet_width.size()):
                jet_graph = self.graph_maker.MakeGraph(self.tree_reader, pj)
                if not (type(jet_graph) is list):
                    print("ERROR: pj =", pj, "entry =", entry, "code =", jet_graph)
                else:
                    layers = ["bar"]
                    jet_graphs.append(jet_graph)

        print("entry", str(nentries) + '/' + str(nentries), "(100%)")
        print("saving to", out_dir, end='\r')

        for i in range(len(self.labels)):
            for j in range(self.n_layers):
                filename = out_dir + 'a' + str(j) + "_tupleGraph_" + self.labels[i] + '.h5'
                dd.io.save(filename, jet_graphs[:][i][j])
                
        print("saved                                           ")
        return