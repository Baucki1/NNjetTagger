import os, sys, subprocess
from subprocess import Popen, PIPE, CalledProcessError

import pickle
import numpy

from CNNjetTagger import CNNjetTagger

class CNNgridSearcher():
    def __init__(self) -> None:
        self.CNN = CNNjetTagger(input_type=True) #only with premade hdf5 reasonable
        
        #default values:
        self.n_folds = 5
        
        self.base_path = "/nfs/dust/atlas/user/bauckhag/code/DESY-ATLAS-BSM/nn_studies/"
        self.submit_template = self.base_path + "submit/gridsearch.submit"
        self.submit_script = self.base_path + "submit/gridsearch.sh"
        
        self.results_dir = self.base_path + "results/gridsearch/"
        
        #default requested computing time
        #gpu request to template
        return

    def __execute(cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)
        return
    
    def start_search(self, data:str, params:dict, input_shape, n_folds=5, batch_size=5000, n_epochs=10, use_gpu=False, h5_blocksize=200000, verbose=0, scoring=None, refit=True, return_train_score=False, return_estimator=False, error_score='raise-deprecating', cv=None, random_state=None, return_n_iter=False, return_times=False) -> None:
        """
        

        Args:
            data (str): HDF5 file directory (no TChains - this makes no sense for a parallelized scan)
            params (dict): dictionary of parameters of model with values to scan
            input_shape (_type_): input shape of images (must match image shapes in input files)
            n_folds (int, optional): # of folds for n-fold scan. Defaults to 5.
            batch_size (int, optional): batch size (must be < h5_blocksize). Defaults to 5000.
            n_epochs (int, optional): number of epochs. Defaults to 10.
            use_gpu (bool, optional): run training on '/device:GPU:0'. Defaults to False.
            h5_blocksize (int, optional): size of dataset blocks in HDF5 input files (must match batch size of input files!). Defaults to 200000.
            verbose (int, optional): _description_. Defaults to 0.
            scoring (_type_, optional): _description_. Defaults to None.
            refit (bool, optional): _description_. Defaults to True.
            return_train_score (bool, optional): _description_. Defaults to False.
            return_estimator (bool, optional): _description_. Defaults to False.
            error_score (str, optional): _description_. Defaults to 'raise-deprecating'.
            cv (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            return_n_iter (bool, optional): _description_. Defaults to False.
            return_times (bool, optional): _description_. Defaults to False.
        """
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        
        n_params = len(params)
        n_param_configs = numpy.prod([len(params[key]) for key in params])
        
        n_jobs = n_folds * n_param_configs
        
        for param_config in range(n_param_configs):
            current_params = {}
            for key in params:
                current_params[key] = params[key][param_config%(params.keys().index(key))]
                param_file = self.results_dir + "params/" + str(param_config) + ".param"
                with open(param_file, 'wb') as file_pi:
                    pickle.dump(current_params, file_pi)
                #param_file_out = open(param_file, "w")
                #param_file_out.write(str(current_params))
                #param_file_out.close()
                for fold in range(n_folds):
                    submit_template = open(self.submit_template, "r")
                    submit_filename = self.base_path + "submit/submit_files/gridsearch_" + str(param_config) + "_" + str(fold) + ".submit"
                    submit_out = open(submit_filename, "w")
                    
                    for line in submit_template:
                        line_out = line
                        if line_out.startswith("Args"):
                            line_out = "Args\t\t= " + data + " " + n_epochs + " " + batch_size + " " + input_shape + " " + param_file + " " + fold + " " + n_folds + " " + use_gpu + " " + h5_blocksize + '\n'
                        if line_out.startswith("Output") or line_out.startswith("Error") or line_out.startswith("Log") or line_out.startswith("Executable"):
                            begin = line_out.split('=')[0].strip()
                            subdir = line_out.split("submit")[1]
                            line_out = begin + "\t\t= " + self.base_path + "submit" + subdir # + script_name + "_" + str(dataset_list.index(dataset)) + "_" + str(split_) + ".out\n"
                        if line_out.startswith("Queue"):
                            line_out = "Queue\t\t" + str(n_folds) + "\n"
                        
                        submit_out.write(line_out)
                        
                    submit_template.close()
                    submit_out.close()
                    
                print("\tsubmitting", submit_filename)
                for line in self.__execute(["condor_submit", submit_filename]):
                    print('\t', line, end="")


    def merge_results(self):
        #search all results files and print out final metrics for each parameter configuration
        metrics = []
        for file in os.listdir(self.results_dir + "results/"):
            if file.endswith(".history"):
                with open(self.results_dir + "results/" + file, "rb") as file_pi:
                    history = pickle.load(file_pi)
                with open(self.results_dir + "params/" + file.split('_')[:-1] + ".param", "rb") as file_pi:
                    params = pickle.load(file_pi)
                print(file, params)
                for key in history:
                    print('\t', key, history[key][-1])
                metrics = list(history.keys())
        
        #search all results files and find best value for each metric
        for metric in metrics:
            best_value = -numpy.inf
            best_params = {}
            for file in os.listdir(self.results_dir + "results/"):
                if file.endswith(".history"):
                    with open(self.results_dir + "results/" + file, "rb") as file_pi:
                        history = pickle.load(file_pi)
                    with open(self.results_dir + "params/" + file.split('_')[:-1] + ".param", "rb") as file_pi:
                        params = pickle.load(file_pi)
                    if history[metric][-1] > best_value:
                        best_value = history[metric][-1]
                        best_params = params
            print(metric, best_value, best_params)
        pass
    

    def check_submit(self, script_name):
        incomplete_jobs = []
        error_jobs, segfaults = [], []
        
        log_path = self.base_path+"submit/log/"
        err_path = self.base_path+"submit/err/"
        
        log_file_template_name = ""
        err_file_template_name = ""
        for line in open(self.base_path+"submit/"+script_name+"_template.submit", "r"):
            if line.startswith("Log"):
                log_file_template_name = line.split('/')[-1].split('$')[0]
            if line.startswith("Error"):
                err_file_template_name = line.split('/')[-1].split('$')[0]
            
        if log_file_template_name == "": 
            print("could not find log file template name")
            
        if err_file_template_name == "":
            print("could not find err file template name")
            
        if len(os.listdir(log_path)) == 0: 
            print("no log files found")
            
        if len(os.listdir(err_path)) == 0:
            print("no err files found")
        
        
        for file in os.listdir(log_path):
            if file.startswith(log_file_template_name):
                log_file = open(log_path + file, "r")
                incomplete = True
                for line in log_file:
                    if "Job terminated of its own accord" in line:
                        incomplete = False
                        break
                if incomplete: incomplete_jobs.append(file)
                log_file.close()
                
        incomplete_jobs.sort()
        print("\n\n" + str(len(incomplete_jobs)), "incomplete jobs:")
        for incomplete_job in incomplete_jobs:
            print("\t" + incomplete_job)
        
        
        for file in os.listdir(err_path):
            if file.startswith(err_file_template_name):
                err_file = open(err_path + file, "r")
                incomplete = False
                try:
                    for line in err_file:
                        if line != "":
                            incomplete = True
                        if line == " *** Break *** segmentation violation\n":
                            segfaults.append(file)
                            break
                except UnicodeDecodeError:
                    #print("unicode error")
                    incomplete = True
                if incomplete: error_jobs.append(file)
                err_file.close()
        
        error_jobs.sort()
        print("\n\n" + str(len(error_jobs)), "errors in jobs found:")
        for error_job in error_jobs:
            print("\t" + error_job)
            
        segfaults.sort()
        print("\n\n" + str(len(segfaults)), "segfaults found:")
        for segfault in segfaults:
            print("\t" + segfault)

# split according to: Params->Cluster, Fold->Process