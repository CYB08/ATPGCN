import torch
import os
from os import listdir
import scipy.io as scio
import numpy as np
from sklearn.metrics import r2_score
from Glasso import GroupLasso
import matplotlib.pyplot as plt
import argparse

GroupLasso.LOG_LOSSES = False
cwd = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--Fun_dir',  type=str, default=os.path.join(cwd, "data/Functional"), help='Functional_direction')
parser.add_argument('--Str_dir',  type=str, default=os.path.join(cwd, "data/Structural"), help='Structural_direction')
parser.add_argument('--Save_dir', type=str, default=os.path.join(cwd, "data/BrainNet"), help='Network_direction')
parser.add_argument('--ROI_risk', type=str, default=os.path.join(cwd, "data/Risk_ROI_List.csv"), help='Risk ROI')
parser.add_argument('--Lambda_group', type=float, default=0.05, help='Lasso Constraint')
parser.add_argument('--Thres', type=float, default=0.01, help='Reserve Reliable Connections')
parser.add_argument('--Vis_Loss', type=bool, default=False, help='Visualize the Regularization')
opt = parser.parse_args()

Funfiles = [f for f in listdir(opt.Fun_dir) if os.path.isfile(os.path.join(opt.Fun_dir, f))];
Funfiles.sort()
Strfiles = []
if not os.path.exists(opt.Str_dir):
    os.makedirs(opt.Str_dir)
else:
    if len(os.listdir(opt.Str_dir))!=0:
        Strfiles = [t for t in listdir(opt.Str_dir) if os.path.isfile(os.path.join(opt.Str_dir, t))];
        Strfiles.sort()

def Compute_grouplasso(A, y, ROItoreg, lam_group, lam_l1=0, Vis_Loss=opt.Vis_Loss):
    ROI_risk=opt.ROI_risk
    if not os.path.isfile(ROI_risk):
        print(ROI_risk + 'is not found!')
        print('ROI grouping information does not exist!')
        ROI_group = np.zeros(116,1)
    else:
        if ROI_risk.endswith('.csv'):
           ROI_group = np.genfromtxt(ROI_risk, dtype=int,
                                    delimiter=',', skip_header=1, usecols=(3)
                                    ).reshape(-1, 1)
    ROI_g = np.delete(ROI_group, ROItoreg, axis=0)
    gl = GroupLasso(groups=ROI_g, group_reg=lam_group, l1_reg=lam_l1,
                    frobenius_lipschitz=True,
                    scale_reg="group_size", subsampling_scheme=1,
                    supress_warning=True, n_iter=1000, tol=1e-3) #scale_reg="inverse_group_size
    gl.fit(A, y)
    yhat = gl.predict(A)
    sparsity_mask = gl.sparsity_mask_
    w_hat = gl.coef_
    R2 = r2_score(y, yhat)
    if Vis_Loss:
        print("Group lasso computing...")
        print(f"Number variables: {len(sparsity_mask)}")
        print(f"Number of chosen variables: {sparsity_mask.sum()}")
        print(f"performance metrics R^2: {R2}")
        plt.figure()
        plt.plot(gl.losses_); plt.title("Loss plot")
        plt.ylabel("Mean squared error"); plt.xlabel("Iteration")
        plt.show()

    return w_hat

def Get_Multimodality_networks(subjects_fun, subjects_str, lam_group=0.1, binary_require = False,
                               variable_fun='ROISignals', variable_str='A',
                               save=True, save_path=opt.Save_dir):
    if not subjects_str:
        subjects_str = [0] * len(subjects_fun)
    for ROI_series, SCNnet in zip(subjects_fun, subjects_str):
        fun_position = os.path.join(opt.Fun_dir, ROI_series)
        timeseries = scio.loadmat(fun_position)[variable_fun]
        NumROI = timeseries.shape[1]
        if SCNnet==0:
            anti_fun = np.ones((NumROI, NumROI))
        else:
            str_position = os.path.join(opt.Str_dir, SCNnet)
            structural_net = scio.loadmat(str_position)[variable_str]
            anti_fun = np.exp(-1*(structural_net**2)/np.std(structural_net))
        w_matrix_list = []
        for j in range(NumROI):
            pen = anti_fun[:,j]
            rp = torch.tensor(pen).repeat(timeseries.shape[0],1).numpy()
            y = timeseries[:,j]
            Fs = np.delete(rp, j, axis=1)
            A = np.delete(timeseries, j, axis=1)/Fs
            w_coeff = Compute_grouplasso(A, y, j, lam_group)
            w_hat_res = np.insert(w_coeff, j, [0])/pen
            w_matrix_list.append(w_hat_res)
        w_matrix = np.array(w_matrix_list, dtype=float)
        if binary_require != True:
            Brain_temp=(w_matrix + w_matrix.T)/2
            Brain_spa_mat = np.where(abs(Brain_temp) > opt.Thres, Brain_temp, 0)
        else:
            Brain_spa_mat = np.int64((w_matrix + w_matrix.T)/2 != 0)

        if save:
            Folder_path = os.path.join(save_path, str(lam_group), 'raw')
            if not os.path.exists(Folder_path):
                os.makedirs(Folder_path)
            scio.savemat(os.path.join(Folder_path, '%s_net_%s.mat' % (ROI_series[:-4], str(lam_group))), {'Brainnetwork':Brain_spa_mat})
            print("===========================")
            print("Save Brain_network: %s_net_%s" % (ROI_series[:-4], str(lam_group)))

    print("===========================")
    print("Multimodal Brain network completed")

def main():
    Get_Multimodality_networks(Funfiles, Strfiles, opt.Lambda_group)

if __name__ == '__main__':
    main()

