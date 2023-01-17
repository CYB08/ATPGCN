# ATPGCN: Adversarially-Trained Persistent Homology-Based Graph Convolutional Network for Disease Identification Using Brain Connectivity
A preliminary version of ATPGCN with demo data which is different from ones in our paper. The example is just used to replicate our framework.


## Setup

### Sparse Brain Network 
We first construct the functional brain connectivity with an open multimodal interface. The method also integrates the ROI-wise group constraint for regularization.   

    Generate_BrainNet_01.py

### Adversarial Example Generation
If it is desired to generate the brain connectome perturbations and perform the adversarial training (optional).

    Generate_Prbs_02.py

### Persistent homology-Based Topology Feature
we extract the persistent homology features of brain conectivity network from an algebraic topology analysis.

    Generate_TopoFeat_03.py

### Model Training and Testing

    Main_04.py 



