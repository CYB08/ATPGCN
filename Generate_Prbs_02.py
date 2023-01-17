import os
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
import argparse
from tqdm import tqdm
from utils.brainnetwork_reader import MyNetworkReader
from torch_geometric.loader import DataLoader
import random
import scipy.sparse as sp

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MultiBrain', help='dataset')
parser.add_argument('--ptb_freq', type=float, default=0.05,  help='pertubation frequency')
parser.add_argument('--label_rate', type=float, default=1, help='rate of labeled data')
parser.add_argument('--nettack_test', type=bool, default=False, help='testing nettack pertubations')
parser.add_argument('--multi_test_poison', type=bool, default=False, help='testing poison attack')
parser.add_argument('--attack_structure', type=bool, default=True, help='nettack structure')
parser.add_argument('--attack_features', type=bool, default=True, help='nettack features')
parser.add_argument('--save_attack', type=bool, default=True, help='Save_attack_if')
parser.add_argument('--batchSize', type=int, default=1000, help='size of the batches')
parser.add_argument('--lam_group', type=float, default=0.05, help='GLasso parameter')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--mat_dir',  type=str, default=os.path.join(os.getcwd(), "data", "Brainnet"), help='Brainnet')
parser.add_argument('--ROI_risk', type=str, default=os.path.join(os.getcwd(), "data", "Risk_ROI_List.csv"), help='Risk ROI')
parser.add_argument('--AP_path',  type=str, default=os.path.join(os.getcwd(), "data", "Brainnet_nettack"), help='Adversarial examples')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


####################Clean brain network read in######################
dataset = MyNetworkReader(os.path.join(args.mat_dir, str(args.lam_group)))
dataset.data.y = dataset.data.y.squeeze()
loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False)
for data in loader:
    adj = sp.csr_matrix((data.edge_attr.squeeze().numpy(),
                         (data.edge_index.numpy()[0, :],
                          data.edge_index.numpy()[1, :])),
                        shape=(data.x.shape[0], data.x.shape[0]),
                        dtype="float32")        # adjacent matrix

    features = sp.csr_matrix((data.x.numpy()))  # group of node features
    ROI_risk = args.ROI_risk
    if not os.path.isfile(ROI_risk):
        print(ROI_risk + 'does not exist!')
    else:
        if ROI_risk.endswith('.csv'):
            ROI_group = np.genfromtxt(ROI_risk,
                                      dtype=int,
                                      delimiter=',',
                                      skip_header=1,
                                      usecols=(3)).reshape(-1, 1)

    ROI_group = ROI_group.squeeze()
    labels = np.tile(ROI_group, int(features.shape[0] / ROI_group.shape[0]))  # group of ROI labels

# Train and Test Split
ROIseries = np.arange(features.shape[0])
train_slice_sub = np.array(random.sample(range(1, ROI_group.shape[0]), 80))
temp = []
# extend to the whole features
for r in range(int(features.shape[0] / ROI_group.shape[0])):
    temp.append(train_slice_sub + r * ROI_group.shape[0])
    train_slice = np.array(temp).reshape(1,-1).squeeze()

test_slice = np.delete(ROIseries, train_slice)
# Setup Surrogate model
train_slice = train_slice[:int(args.label_rate * adj.shape[0])]
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, train_slice, patience=30)

def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, train_slice, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def test(adj, features, target_node):
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, train_slice, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[test_slice], labels[test_slice])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def main():
    risk_nodes_indices = np.append(np.where((labels == 1))[0],
                                   np.where((labels == 2))[0])
    #ptb_num = int(args.ptb_freq * len(risk_nodes_indices))
    ptb_num = 3
    att_nodes_list = random.sample(risk_nodes_indices.tolist(),ptb_num)
    modified_adj = adj
    Perstru_res = adj
    Perfeatures_res = features
    num = len(att_nodes_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    cnt = 0
    perturbations_feat = []
    perturbations_stru = []
    for target_node in tqdm(att_nodes_list):
        n_perturbations = np.random.randint(40, 50)
        model = Nettack(surrogate, nnodes=adj.shape[0],
                        attack_structure=args.attack_structure,
                        attack_features=args.attack_features,
                        device=device)

        model = model.to(device)
        model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
        modified_features = model.modified_features
        modified_adj = model.modified_adj
        if args.attack_features:
            feature_perturbations = model.feature_perturbations
            [perturbations_feat.append(x) for x in feature_perturbations if ((x not in perturbations_feat) and (len(x)!=0))]
        if args.attack_structure:
           structure_perturbations = model.structure_perturbations
           for y in structure_perturbations:
               if y not in perturbations_stru and len(y)!=0:
                   perturbations_stru.append(y)

        if args.nettack_test:
            print('=== testing GCN on original(clean) graph ===')
            test(adj, features, target_node)
            print('=== testing GCN on perturbed graph ===')
            test(modified_adj, modified_features, target_node)

        if args.multi_test_poison:
            acc = single_test(modified_adj, modified_features, target_node)
            if acc == 0:
                cnt += 1

    #print('misclassification rate : %s' % (cnt/num))
    if args.save_attack:
        if args.attack_features:
            for per_ix in perturbations_feat:
                if len(per_ix)!=0:
                    Perfeatures_res[per_ix] = 1 - Perfeatures_res[per_ix]

        if args.attack_structure:
            for per_iy in perturbations_stru:
                if len(per_iy) != 0:
                    Perstru_res[per_iy] = Perstru_res[per_iy[::-1]] = 1 - Perstru_res[per_iy]

        path = os.path.join(args.AP_path, str(args.lam_group), str(ptb_num))  # args.ptb_freq
        if not os.path.exists(path):
            os.makedirs(path)
        if args.attack_features:
            features_path = os.path.join(path, "{}_nettack_fea.npz".format(args.dataset))
            sp.save_npz(features_path, Perfeatures_res)
        if args.attack_structure:
            adj_path = os.path.join(path, "{}_nettack_adj.npz".format(args.dataset))
            sp.save_npz(adj_path, Perstru_res)

        print("Adversarial Example Saved")

if __name__ == '__main__':
    main()

