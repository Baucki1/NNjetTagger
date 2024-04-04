class CaloSampling():
    def __init__(self):
        self.Names = ["PreSamplerB", "EMB1", "EMB2", "EMB3", 
                      "TileBar0", "TileBar1", "TileBar2", 
                      "TileGap1", "TileGap2", "TileGap3", 
                      "TileExt0", "TileExt1", "TileExt2", 
                      "PreSamplerE", "EME1", "EME2", "EME3", 
                      "HEC0", "HEC1", "HEC2", "HEC3", 
                      "FCAL0", "FCAL1", "FCAL2"]
        
        self.index2Name_dict ={0 : "PreSamplerB", 1 : "EMB1",      2 : "EMB2",   3 : "EMB3", 
             4 : "PreSamplerE", 5 : "EME1",      6 : "EME2",   7 : "EME3", 
             8 : "HEC0",        9 : "HEC1",     10 : "HEC2",  11 : "HEC3", 
             12 : "TileBar0",  13 : "TileBar1", 14 : "TileBar2", 
             15 : "TileGap1",  16 : "TileGap2", 17 : "TileGap3", 
             18 : "TileExt0",  19 : "TileExt1", 20 : "TileExt2", 
             21 : "FCAL0",     22 : "FCAL1",    23 : "FCAL2", 
             24 : "MINIFCAL0", 25 : "MINIFCAL1", 26 : "MINIFCAL2", 27 : "MINIFCAL3", 
             28 : "Unknown"}

        self.index2Region_dict ={0 : "bar", 1 : "bar", 2 : "bar", 3 : "bar", 12 : "bar", 13 : "bar", 14 : "bar",
                                 4 : "end", 5 : "end", 6 : "end", 7 : "end", 8 : "end", 9 : "end", 10 : "end", 11 : "end",  
                                 18 : "ext", 19 : "ext", 20 : "ext"}

        self.Name2Layer_dict = {"PreSamplerB" : 0, "EMB1" : 0, "EMB2" : 0, "EMB3" : 0 , 
                                "TileBar0" : 1, "TileBar1" : 2, "TileBar2" : 3,
                                "TileExt0" :  0, "TileExt1" :  1, "TileExt2" :  2,
                                "PreSamplerE" : 0 , "EME1" :  0, "EME2" :  0, "EME3" :  0, 
                                "HEC0" :  1, "HEC1" :  2, "HEC2" :  3, "HEC3" :  4}
        
        self.allowedLayers = self.Name2Layer_dict.keys()

    

    def index2Name(self, index):
        return self.index2Name_dict[index]

    def index2Region(self, index):
        return self.index2Region_dict[index]

    def name2Layer(self, name):
        return self.Name2Layer_dict[name]

