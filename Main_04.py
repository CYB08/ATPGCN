import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from utils.brainnetwork_reader import MyNetworkReader
from torch_geometric.data import DataLoader
from model.gnn import MyGCN
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from utils.attacked_data import MyReadPtbData
from retrying import retry
import torch.nn as nn
import copy
import deepdish as dd
import Generate_TopoFeat_03 as generate_external_topofeat

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--mat_dir', type=str, default=os.path.join(os.getcwd(), "data", "Brainnet"), help='BrainNet_dir')
parser.add_argument('--adv_dir', type=str, default=os.path.join(os.getcwd(), "data", "Brainnet_nettack"), help='APs_dir')
parser.add_argument('--topo_dir', type=str, default=os.path.join(os.getcwd(), "data", "TopoFeat"), help='Topology_feature_dir')
parser.add_argument('--lam_group', type=float, default=0.05, help='lam_group')
parser.add_argument('--CV_splits', type=int, default=10, help='CV_splits')
parser.add_argument('--val_slice', type=float, default=0.8, help='val_slice')
parser.add_argument('--train_with_val', type=bool, default=True, help='If performing val')
parser.add_argument('--numofROI', type=int, default=116, help='numofROI')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=100, help='batche size')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler')
parser.add_argument('--weightdecay', type=float, default=5e-4, help='regularization')
parser.add_argument('--infeat', type=int, default=116, help='in_feature_dim of GCN')
parser.add_argument('--outfeat', type=int, default=32, help='out_feature_dim of GCN')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--attack', type=str, default='no', choices=['no','nettack'])
parser.add_argument('--ptb_rate', type=float, default=10, help="noise ptb_rate")
parser.add_argument('--dataset', type=str, default="MultiBrain", help='dataset')
parser.add_argument('--land_layers', type=int, default=8, help='land')
parser.add_argument('--lasland', type=int, default=32, help='lasland')
parser.add_argument('--resolution', type=int, default=1000, help='Resolution for landscapes')
parser.add_argument('--lamb_1', type=float, default=0.3, help='PH-1 LOSS')
parser.add_argument('--lamb_2', type=float, default=0.4, help='PH-0 LOSS')
parser.add_argument('--lamb_3', type=float, default=0.15, help='VAE LOSS')
parser.add_argument('--lamb_4', type=float, default=0, help='L2 regularization')
opt = parser.parse_args()

#################### Parameter Initialization #######################
lam_group = opt.lam_group
val_slice = opt.val_slice
land = opt.land_layers
resolution = opt.resolution
CV_splits = opt.CV_splits
numofROI = opt.numofROI
lasland = opt.lasland
train_with_val = opt.train_with_val
loss_min = 1e5
topo_dir = opt.topo_dir
adv_dir = opt.adv_dir
BrainNetwork_dir = os.path.join(opt.mat_dir, str(lam_group))
num_epoch = opt.n_epochs
model_parameters_shw = False


############################### Define Loss Functions ########################################
def distance_win_loss(land_group):
    Within_Group = torch.norm(land_group[:, None] - land_group, dim=2, p=2)**2
    pho = Within_Group.shape[0] * (Within_Group.shape[0]-1) / 2
    dis_win_loss = torch.sum(torch.triu(Within_Group, diagonal = 1))/pho
    return dis_win_loss

def distance_wout_loss(cn_group, pat_group):
    Without_Group = torch.norm(cn_group[:, None] - pat_group, dim=2, p=2)**2
    dis_wout_loss = torch.mean(Without_Group)
    return dis_wout_loss

def reconstruct_loss(recon_x, x, mu, logvar):
    recons_loss = F.mse_loss(recon_x, x)
    kld_loss = torch.mean(-0.5 * torch.sum(
        1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return recons_loss + kld_loss

def land_loss(land_embed, la_mu, la_logvar, land_original, la_decoder, y, dis=True, pear=True):
    if land_embed.shape[0]!=y.shape[0]:
        print("Warning: the number of subjects is incorrect in landscapes")
    else:
        Sub_cn  = la_mu[(y == 0).nonzero().flatten(), :]
        Sub_pat = la_mu[(y == 1).nonzero().flatten(), :]
        L_sub = distance_win_loss(Sub_cn) + distance_win_loss(Sub_pat)
        #L_group = -distance_wout_loss(Sub_cn, Sub_pat)
        L_con = reconstruct_loss(la_decoder, land_original, la_mu, la_logvar)
    return L_sub, L_con

def BettiCurve_loss(betti, y):
    Sub_cn = betti[(y == 0).nonzero().flatten(), :]
    Sub_pat = betti[(y == 1).nonzero().flatten(), :]
    L_sub = distance_win_loss(Sub_cn) + distance_win_loss(Sub_pat)
    L_group = -distance_wout_loss(Sub_cn, Sub_pat)
    return L_sub + 0.33 * L_group

def L2Loss(model,alpha):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss


###################### Network Training Function ##################################
def train_begin(External_fea):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_all = 0
    step = 0
    seq_idx = 0
    train_correct = []
    Landscapes = External_fea[0]
    DervCurves = External_fea[1]
    for data in train_loader:
        data = data.to(device)
        PH1_feat = Landscapes[seq_idx:data.num_graphs+seq_idx]
        PH0_feat = DervCurves[seq_idx:data.num_graphs+seq_idx]
        seq_idx = seq_idx + data.num_graphs
        optimizer.zero_grad()
        output, la_mu, la_logvar, land_embed, la_decoder, Betti= model(data.x, data.edge_index, data.batch,
                                                                       data.edge_attr, PH1_feat, PH0_feat)
        loss_c = criterion(output, data.y)
        loss_ph1, loss_vae = land_loss(land_embed, la_mu, la_logvar, PH1_feat, la_decoder, data.y)
        loss_ph0 = BettiCurve_loss(Betti, data.y)
        loss = loss_c + opt.lamb_1 * loss_ph1 + opt.lamb_2 * loss_ph0 + \
               opt.lamb_3 * loss_vae + opt.lamb_4 * L2Loss(model,0.1)
        step = step + 1
        if (torch.cuda.is_available()):
            loss = loss.cuda()
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
        _, train_pred = torch.max(output, 1)
        train_correct.append((train_pred == data.y).sum().item())
        scheduler.step()
    return loss_all / len(train_loader.dataset), sum(train_correct) / len(train_loader.dataset)

@retry(stop_max_attempt_number=1)
def test_begin(loader, External_fea, state):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_all = 0
    AUC = 0
    Sensitivity = 0
    Specificity = 0
    seq_idx = 0
    test_correct = []
    Landscapes = External_fea[0]
    DervCurves = External_fea[1]
    for data in loader:
        data = data.to(device)
        PH1_feat =  Landscapes[seq_idx:data.num_graphs+seq_idx]
        PH0_feat =  DervCurves[seq_idx:data.num_graphs+seq_idx]
        seq_idx = seq_idx+data.num_graphs
        output, la_mu, la_logvar, land_embed, la_decoder, Betti = model(data.x, data.edge_index, data.batch,
                                                                        data.edge_attr, PH1_feat, PH0_feat)
        if state in ['val']:
            loss_c = criterion(output, data.y)
            loss_ph1, loss_vae = land_loss(land_embed, la_mu, la_logvar, PH1_feat, la_decoder, data.y)
            loss_ph0 = BettiCurve_loss(Betti, data.y)
            loss = loss_c + opt.lamb_1 * loss_ph1 + opt.lamb_2 * loss_ph0 + \
                   opt.lamb_3 * loss_vae + opt.lamb_4 * L2Loss(model, 0.1)
            loss_all += loss.item() * data.num_graphs
            _, test_pred = torch.max(output, 1)
            test_correct.append((test_pred == data.y).sum().item())
        elif state in ['test']:
            _, test_pred = torch.max(output, 1)
            test_correct.append((test_pred == data.y).sum().item())
            cfm = confusion_matrix(data.y, test_pred)
            AUC = metrics.roc_auc_score(data.y, output[:, 1])
            tn_sum = cfm[0, 0]  # True Negative
            fp_sum = cfm[0, 1]  # False Positive
            tp_sum = cfm[1, 1]  # True Positive
            fn_sum = cfm[1, 0]  # False Negative
            Condition_negative_sen = tp_sum + fn_sum
            Sensitivity = tp_sum / Condition_negative_sen
            Condition_negative_spe = tn_sum + fp_sum
            Specificity = tn_sum / Condition_negative_spe

    return loss_all / len(loader.dataset), sum(test_correct) / len(loader.dataset), sum(test_correct), Sensitivity, Specificity, AUC


    ###################### Topological features ##########################
@retry(stop_max_attempt_number=3)
def Read_External_Topologyfeas(dir, tr_idx, va_idx, te_idx):
    ######################## PH characterization ########################
    PH_0_list = []
    PH_1_list = []
    External_Topologyfeas = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))];
    External_Topologyfeas.sort()
    if len(External_Topologyfeas)!= len(Sub_list):
        print('#######################################')
        print('not find the topological feature ')
        print('#######################################')
        print('Executing topological feature generation module.... ')
        ############# Execting topological feature extraction ##############
        generate_external_topofeat.main()
        print('Completed!')
        raise IOError("Extracting")
    else:
        for idx, res in enumerate(External_Topologyfeas):
            topolfeas_file = dd.io.load(os.path.join(dir, res))
            # read edge and edge attribute
            PH_0 = topolfeas_file['Derivative_curves'][()]
            PH_1 = topolfeas_file['Persistence_landscapes'][()]
            PH_0_list.append(PH_0)
            PH_1_list.append(PH_1)

        DevCurve_temp = torch.stack(PH_0_list)
        PLand_temp  = torch.stack(PH_1_list)
        PH1_TR  = torch.index_select(PLand_temp,   0, torch.LongTensor(tr_idx))
        PH1_VAR = torch.index_select(PLand_temp,   0, torch.LongTensor(va_idx))
        PH1_TE  = torch.index_select(PLand_temp,   0, torch.LongTensor(te_idx))
        PH0_TR  = torch.index_select(DevCurve_temp, 0, torch.LongTensor(tr_idx))
        PH0_VAR = torch.index_select(DevCurve_temp, 0, torch.LongTensor(va_idx))
        PH0_TE  = torch.index_select(DevCurve_temp, 0, torch.LongTensor(te_idx))
        External_fea_TR  = [PH1_TR, PH0_TR]
        External_fea_VAR = [PH1_VAR, PH0_VAR]
        External_fea_TE  = [PH1_TE, PH0_TE]

    return External_fea_TR, External_fea_VAR, External_fea_TE


######################### Define Dataloader ##########################
if __name__ == '__main__':
    dataset = MyNetworkReader(BrainNetwork_dir)
    Sub_list = dataset.subject
    ######################## insert perturbation ########################
    if opt.attack == 'no':
        print('processing under non-perturbation')
        dataset_perturbation = dataset
    if opt.attack == 'nettack': #nettack
        dataset_perturbation = copy.deepcopy(dataset)
        perturbed_data = MyReadPtbData(root=os.path.join(adv_dir, str(lam_group), str(opt.ptb_rate)),
                                     name=opt.dataset,
                                     ROI_num=numofROI,
                                     attack_method=opt.attack,
                                     ptb_rate=opt.ptb_rate)

        dataset_perturbation.data.x = perturbed_data.features
        dataset_perturbation.data.edge_attr = perturbed_data.edge_attr
        dataset_perturbation.data.edge_index = perturbed_data.edge_index
        dataset_perturbation.data.y = dataset_perturbation.data.y.squeeze()

    dataset.data.y = dataset.data.y.squeeze()
    skf = StratifiedKFold(n_splits = CV_splits, shuffle=True, random_state=0)
    valid_idx = None
    total_samples = 0
    total_acc = 0; total_auc = 0
    total_sen = 0; total_spe = 0
    for train_idx, test_idx in skf.split(dataset, dataset.data.y):
        #################### Dataset slicing #####################
        if train_with_val:
            train_size = int(val_slice * len(train_idx))
            valid_size = len(train_idx) - train_size
            train_idx, valid_idx = torch.utils.data.random_split(train_idx, [train_size, valid_size])
            val_dataset = dataset_perturbation[torch.LongTensor(valid_idx).to(device)]
            val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
        train_dataset = dataset_perturbation[torch.LongTensor(train_idx).to(device)]
        train_loader  = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=False)
        test_dataset  = dataset[torch.LongTensor(test_idx).to(device)]
        test_loader   = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

        ############### External topological feature of individual-level subject ###############
        topofea_TR, topofea_VAR, topofea_TE = Read_External_Topologyfeas(topo_dir, train_idx, valid_idx, test_idx)

        ###############################  GCN Model instancing  #################################
        model = MyGCN(opt.infeat, opt.outfeat, opt.nclass, numofROI,
                      topofea_TR[0].shape[1], topofea_TR[1].shape[1]).to(device)  #infeat=116, outfeat=32, nclass=2
        if model_parameters_shw is not None:
            par_ = print([param_opt for param_opt, tem in model.named_parameters() if tem.requires_grad]) # learnable parameters
        optimizer = torch.optim.Adam(model.parameters(), lr= 0.01, weight_decay=opt.weightdecay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) #gamma=0.5

        ################################  Model Training #################################
        model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())
        since = time.time()
        for epoch in range(0, num_epoch):
            tr_loss, tr_acc = train_begin(topofea_TR)
            ################ Training without CV ################
            if valid_idx is None:
                if epoch % 1 == 0:
                    with torch.no_grad():
                        te_res = test_begin(test_loader, topofea_TE, state='val')
                        val_loss_epoch = te_res[0]
                        val_acc_epoch  = te_res[1]
                    print('~~~~~~~~~~~~~~~')
                    print('Epoch: %d trainloss: %.5f testloss: %.5f train_accuracy: %.5f %% test_accuracy: %.5f %%' %
                          (epoch + 1, tr_loss, val_loss_epoch, 100 * tr_acc, 100 * val_acc_epoch))
            ################# Training with CV ################
            else:
                if epoch % 1 == 0:
                    with torch.no_grad():
                        te_res = test_begin(val_loader, topofea_VAR, state='val')
                        val_loss_epoch = te_res[0]
                        val_acc_epoch  = te_res[1]
                    print('~~~~~~~~~~~~~~~')
                    print('Epoch: %d trainloss: %.5f testloss: %.5f train_accuracy: %.5f %% test_accuracy: %.5f %%' %
                          (epoch + 1, tr_loss, val_loss_epoch, 100 * tr_acc, 100 * val_acc_epoch))

                if  val_loss_epoch < loss_min and epoch > 10:
                    loss_min = val_loss_epoch
                    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Finished Training in {:.0f}m {:.0f}s'.
              format(time_elapsed//60, time_elapsed % 60))

        ############################  Model Testing  ##############################
        with torch.no_grad():
            if valid_idx:
                model.load_state_dict(model_wts_lowest_val_loss)
            loss, acc, corrects, sen, spe, auc = test_begin(test_loader, topofea_TE, state='test')
            test_samples = len(test_loader.dataset)
            total_samples = total_samples + test_samples
            total_acc = total_acc + corrects
            total_auc = total_auc + auc
            total_sen = total_sen + sen
            total_spe = total_spe + spe

            print("========== Model testing ==========")
            print('\033[1;33mAccuracy: %.5f %%\033[0m' % (100 * acc))
            print('\033[1;32mSensitivy: %.5f %%\033[0m' % (100 * sen))
            print('\033[1;34mSpecificity: %.5f %%\033[0m' % (100 * spe))
            print('\033[1;36mAUC: %.5f\033[0m' % (auc))

    print("========== The test results ==========")
    print('\033[4;30mTotal Accuracy: %.5f %%\033[0m' % (100 * total_acc / total_samples))
    print('\033[4;30mTotal Sensitivy: %.5f %%\033[0m' % (100 * total_sen / CV_splits))
    print('\033[4;30mTotal Specificity: %.5f %%\033[0m' % (100 * total_spe / CV_splits))
    print('\033[4;30mTotal AUC: %.5f\033[0m' % (total_auc / CV_splits))
