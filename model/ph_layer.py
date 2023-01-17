import torch
import torch.nn as nn
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations
from sklearn_tda import *
import scipy.sparse as sp
from numpy import polynomial as P
import warnings
warnings.filterwarnings("ignore")
#warnings.simplefilter('ignore', np.RankWarning)

class MyPHlayer(nn.Module):
    def __init__(self, resolution, land, dim_obj, nROI):
        super(MyPHlayer, self).__init__()
        self.resolution = resolution
        self.nROI = nROI
        self.nland = land
        self.sampling_control = 50
        self.dim_obj = dim_obj   # 0-dimensional and 1-dimensional persistent feature

    def forward(self, x, edge_index, edge_attr):
        land, betti, curve = self.Compute_Persistent_Feats(x, edge_index, edge_attr,self.resolution, self.nland,self.dim_obj)
        land_PH1_feats  = torch.from_numpy(np.squeeze(np.array(land,dtype = np.float32)))
        betti_feats = torch.from_numpy(np.squeeze(np.array(betti,dtype = np.float32)))
        curve_PH0_feats = torch.from_numpy(np.squeeze(np.array(curve,dtype=np.float32)))

        return land_PH1_feats, betti_feats, curve_PH0_feats

    def extract_block_diag(self, A, M, k=0):
        """Extracts blocks of size M=numofROIs from the kth diagonal of brain connectivity A,
        whose size must be a multiple of M."""
        # Check that the matrix can be block divided
        if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
            print('Matrix must be square and a multiple of block size')
        # Assign indices for offset from main diagonal
        if abs(k) > M - 1:
            print('kth diagonal does not exist in matrix')
        elif k > 0:
            ro = 0
            co = abs(k) * M
        elif k < 0:
            ro = abs(k) * M
            co = 0
        else:
            ro = 0
            co = 0

        blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M]
                           for i in range(0,len(A)-abs(k)*M,M)])
        return blocks


    def Compute_Persistent_Feats(self, x, edge_index, edge_attr, res, nland, dim_obj):
        adj_gro = sp.coo_matrix(
            (edge_attr.squeeze().numpy(), (edge_index.numpy()[0, :], edge_index.numpy()[1, :])),
            shape=(x.shape[0], x.shape[0]), dtype="float32").toarray()

        adj_sub = self.extract_block_diag(adj_gro, self.nROI)

        Land_feature = []
        betti_num = []
        curve_feature = []
        Digm_interval = []
        skeleton = []
        Landscapes_shw= None
        Betti_shw = None
        for i in range(adj_sub.shape[0]):
            correlation_matrix = np.array(adj_sub[i], float)
            #distance_matrix = np.sqrt(1-correlation_matrix**2)
            distance_matrix = 1-correlation_matrix
            rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=10)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            for sk_value in simplex_tree.get_skeleton(1):
                skeleton.append(tuple(sk_value))
            Diags = simplex_tree.persistence(homology_coeff_field=11, min_persistence=0)
            PH_0 = simplex_tree.persistence_intervals_in_dimension(0)
            PH_1 = simplex_tree.persistence_intervals_in_dimension(1)

            if dim_obj == 0:
                ######################### Betti-0 curves #########################
                diags_dim0 = DiagramSelector(use=True, point_type="finite").\
                    fit_transform([PH_0])
                BC = BettiCurve(resolution=res)
                Betti_num_curve = BC.fit_transform(diags_dim0)
                X_axis = np.arange(res).reshape(-1, 1).squeeze()
                p = P.polynomial.Polynomial.fit(X_axis, Betti_num_curve.squeeze(), deg=500)
                yvals = p(X_axis)  # fitting curve
                fderiv = abs(p.deriv()(X_axis))[self.sampling_control:(res - self.sampling_control)]  # derivative curve
                betti_num.append(yvals)
                curve_feature.append(fderiv)
                if Betti_shw is not None:
                    fig, (ax1, ax2) = plt.subplots(2, 1)
                    ax1.plot(yvals)
                    ax2.plot(fderiv)
                    plt.show()

                ############################## Landscapes ############################
                diags_dim1 = DiagramSelector(use=True, point_type="finite").\
                    fit_transform([PH_1])
                LS = gd.representations.Landscape(num_landscapes = nland, resolution = res)  # landscape computing
                L = LS.fit_transform(diags_dim1)
                land_average = np.average(L.reshape(nland, res), 0)
                Land_feature.append(land_average)
                Digm_interval.append(diags_dim1)
                if Landscapes_shw is not None:
                    for i in range(nland):
                        plt.plot(L[0][i * 1000:(i + 1) * 1000], color='silver',linestyle='-.', linewidth=3)
                        plt.plot(land_average, color='red',  marker='*', markersize=2)
                        plt.tick_params(labelsize=25)
                        plt.title("Individual Landscape", fontsize='xx-large')
                    plt.show()

        return Land_feature, betti_num, curve_feature