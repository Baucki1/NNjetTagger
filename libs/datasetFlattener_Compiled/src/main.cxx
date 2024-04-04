#include "T.h"
#include "TSystem.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1D.h"
#include "TROOT.h"
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include "TSystemDirectory.h"
#include "TList.h"
#include <sys/stat.h>
#include <chrono>

bool is_file(const char *path)
{
    struct stat buf;
    stat(path, &buf);
    return S_ISREG(buf.st_mode);
}

bool is_dir(const char *path)
{
    struct stat buf;
    stat(path, &buf);
    return S_ISDIR(buf.st_mode);
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        std::cerr << "Usage: runAnalysis PATH_TO_INPUT OUT_FILE_PATH" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile_path = argv[2];

    std::cout << "Writing to dir: " << outputFile_path << std::endl;

    std::vector<TString> namefileslist;
    TChain *tree = new TChain("T");

    if (is_file(inputFile.c_str()))
    {
        std::cerr << "Adding file " << inputFile << std::endl;
        namefileslist.push_back(inputFile);
        tree->Add(TString(inputFile));
        std::cout << inputFile << " " << tree->GetEntriesFast() << std::endl;
    }
    else if (is_dir(inputFile.c_str()))
    {
        std::cerr << "Opening directory " << inputFile << std::endl;
        if (inputFile.substr(inputFile.length() - 1) != "/")
            inputFile = inputFile + "/";
        TSystemDirectory *directory = new TSystemDirectory(inputFile.c_str(), inputFile.c_str());
        TList *files = directory->GetListOfFiles();
        if (files)
        {
            TSystemFile *file;
            TString fname;
            TIter next(files);
            while ((file = (TSystemFile *)next()))
            {
                fname = file->GetName();
                if (!file->IsDirectory() && fname.EndsWith(".root"))
                {
                    std::cerr << "  adding file " << fname << std::endl;
                    namefileslist.push_back(fname);
                    tree->Add(TString(inputFile + fname));
                }
            }

            delete file;
            delete files;
            delete directory;
        }
    }


    std::string outputFile_name = outputFile_path + "/out.root";
    std::cout << "Output file name: " << outputFile_name << std::endl;
    std::cerr << "Processing ..." << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    T *t = new T(tree);
    t->Loop(outputFile_name);
    delete t;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cerr << "Event loop took " << int_ms.count() << " ms" << std::endl;
    std::cerr << "All done!" << std::endl;

    return 0;
}
