class TreeHolder():
        
    def __init__(self, T=None):
        self.set_tree(T)
        
    def set_tree(self, T):
        self.TTree = T
        
    def update(self):
        self.types = self.TTree.types
        
        self.jet_index = self.TTree.LJjet_index        
        self.jet_gapRatio = self.TTree.GNNjet_gapRatio
        self.jet_width = self.TTree.GNNjet_width
        self.jet_IsBIB = self.TTree.LJjet_IsBIB
    
        self.jet_clusEta = self.TTree.GNNjet_clusEta
        self.jet_clusPhi = self.TTree.GNNjet_clusPhi
        self.jet_clusESampl = self.TTree.GNNjet_clusESampl
        self.jet_clusSamplIndex = self.TTree.GNNjet_clusSamplIndex
        