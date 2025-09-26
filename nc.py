from utils.data_manager import DataManager
from utils.nc_utils import AverageMeter, compute_accuracy, print_and_save
from utils import factory
import scipy.linalg as scilin
import numpy as np
import argparse
import pickle
import torch
import json
import copy
import os 

HARD_DRIVE_PATH = '/media/data/ctsiggiropoulos/' 
INFO_FILE_NAME = 'model_on_task_9.pkl'

TOTAL_CLASSES = 100
CIFAR100_TRAIN_SAMPLES = 100 * (500,)
CIFAR100_TEST_SAMPLES = 100 * (100,)

batch_size = 128
num_workers = 8
STORE_DATA_ONLY_FOR_THE_NEW_TASK = False
prefix = 'new_' if STORE_DATA_ONLY_FOR_THE_NEW_TASK else ''


class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []



def main(args, main_path): 
    path = HARD_DRIVE_PATH + main_path + '/gamma=%s_kd_gamma=%s_seed=%d/' % (str(args['gamma']).replace('.', "_"), str(args['kd_gamma']).replace('.', "_"), args['seed'])
    # check if necessary files exist 
    for i, _ in enumerate(range(args['init_cls'], TOTAL_CLASSES, args['increment'])):
        if not os.path.exists(path + 'model_on_task_{}.pkl'.format(i)): 
            print("[WARNING] - Necessary Files Missing >>> The following file doesn't exist : '" + path + 'model_on_task_{}.pkl'.format(i) + "'")
            return
    # if os.path.exists(path + prefix + 'info.pkl') : 
    #     print(">>> [WARNING] FILE ALREADY EXISTS FOR MODEL %s, g=%.2f, kd=%.2f, s=%d" % (args['model_name'], args['gamma'], args['kd_gamma'], args['seed'])) 
    #     print("     Going to next model ...", end='\n\n') 
    #     return
    
    print(">>> [INFO] Working on model %s, g=%.2f, kd=%.2f, s=%d" % (args['model_name'], args['gamma'], args['kd_gamma'], args['seed']))

    fc_features = FCFeatures()
    info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'W': [],
        'b': [],
        'H': [],
        'mu_G_train': [],
        'mu_G_test': [],
        'train_acc1': [],
        'train_acc5': [],
        'test_acc1': [],
        'test_acc5': [],
        'trace_sigma_w': [],
        'trace_sigma_b': [],
        'collapse_metric_test': [],
        'WH_relation_metric_test': [],
        'Wh_b_relation_metric_test': [],
    }
    _device = torch.device('cuda:{}'.format(args['device'][0]))

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1
    )

    model = factory.get_model(args["model_name"], args)
    fc_features = FCFeatures()
    _previous_classes = 0

    for task in range(data_manager.nb_tasks):
        # Get datasets for computing neural collapse
        _total_classes = task*args['increment'] + args['init_cls']

        # MAKE DATASET 
        if STORE_DATA_ONLY_FOR_THE_NEW_TASK : 
            train_dataset = data_manager.get_dataset(np.arange(_previous_classes, _total_classes), source="train", mode="test")
            test_dataset = data_manager.get_dataset(np.arange(_previous_classes, _total_classes), source="test", mode="test")
        else : 
            train_dataset = data_manager.get_dataset(np.arange(0, _total_classes), source="train", mode="test")
            test_dataset = data_manager.get_dataset(np.arange(0, _total_classes), source="test", mode="test")
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 

        # get saved model from the file 
        checkpoint = torch.load(path + 'model_on_task_{}.pkl'.format(task))
        # update fc head 
        model._network.update_fc(_total_classes)
        # update model with the one from the file
        model._network.load_state_dict(checkpoint['model_state_dict'])
        model._network.to(_device)
        handle = model._network.fc.register_forward_pre_hook(fc_features) 
        model._network.eval()

        # RUN 
        for n, p in model._network.named_parameters():
            if 'fc.weight' in n:
                W = p
            if 'fc.bias' in n:
                b = p

        _previous_classes = 0 if not STORE_DATA_ONLY_FOR_THE_NEW_TASK else _previous_classes
        mu_G_train, mu_c_dict_train, train_acc1, train_acc5 = compute_info(_device, model._network, fc_features, trainloader, _previous_classes,  _total_classes, isTrain=True)
        mu_G_test, mu_c_dict_test, test_acc1, test_acc5 = compute_info(_device, model._network, fc_features, testloader, _previous_classes, _total_classes, isTrain=False)
        Sigma_W = compute_Sigma_W(_device, model._network, fc_features, mu_c_dict_train, trainloader, _previous_classes, _total_classes, isTrain=True)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train, _previous_classes)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)

        WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train, _previous_classes)
        # WITHOUT BIAS TERM (b=torch.zeros())
        Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train)

        if args.get('save_test_metrics', False) : 
            # FOR TEST SET 
            sigma_w_test = compute_Sigma_W(_device, model._network, fc_features, mu_c_dict_test, testloader, _previous_classes, _total_classes, isTrain=False)
            sigma_b_test = compute_Sigma_B(mu_c_dict_test, mu_G_test, _previous_classes)

            collapse_metric_test = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
            # ETF DOESN'T CHANGE 
            WH_relation_metric_test, H_test = compute_W_H_relation(W, mu_c_dict_test, mu_G_test, _previous_classes)
            Wh_b_relation_metric_test = compute_Wh_b_relation(W, mu_G_test)

            info_dict['collapse_metric_test'].append(collapse_metric_test)
            info_dict['WH_relation_metric_test'].append(WH_relation_metric_test)
            info_dict['Wh_b_relation_metric_test'].append(Wh_b_relation_metric_test)
            
            info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())


        info_dict['trace_sigma_w'].append(np.trace(Sigma_W)) 
        info_dict['trace_sigma_b'].append(np.trace(Sigma_B)) 

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric)
        info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

        info_dict['W'].append((W.detach().cpu().numpy()))
        info_dict['H'].append(H.detach().cpu().numpy())

        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)

        _printable = '[Tasks: %d/%d] | train top1: %.4f | train top5: %.4f | test top1: %.4f | test top5: %.4f ' % (task+1, data_manager.nb_tasks, train_acc1, train_acc5, test_acc1, test_acc5) 
        print(_printable)        
        _previous_classes = _total_classes

    with open(path + prefix + 'info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/lwf.json',
                        help='Json file of settings.')
    return parser

def compute_info(device, model, fc_features, dataloader, previous_n_classes, n_classes, isTrain=True, dataset='cifar100'):
    mu_G = 0
    mu_c_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (_, inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)['logits']

        features = fc_features.outputs[0][0]
        fc_features.clear()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] += features[b, :]

        prec1, prec5 = compute_accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        

    if dataset == 'cifar100':
        if isTrain:
            mu_G /= sum(CIFAR100_TRAIN_SAMPLES[previous_n_classes:n_classes])
            for i in range(previous_n_classes, n_classes):
                mu_c_dict[i] /= CIFAR100_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(CIFAR100_TEST_SAMPLES[previous_n_classes:n_classes])
            for i in range(previous_n_classes, n_classes):
                mu_c_dict[i] /= CIFAR100_TEST_SAMPLES[i]

    return mu_G, mu_c_dict, top1.avg, top5.avg

def compute_Sigma_W(device, model, fc_features, mu_c_dict, dataloader, previous_n_classes, n_classes, isTrain=True, dataset='cifar100'):

    Sigma_W = 0
    for batch_idx, (_, inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)['logits']

        features = fc_features.outputs[0][0]
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if dataset == 'cifar100':
        if isTrain:
            Sigma_W /= sum(CIFAR100_TRAIN_SAMPLES[previous_n_classes:n_classes])
        else:
            Sigma_W /= sum(CIFAR100_TEST_SAMPLES[previous_n_classes:n_classes])

    return Sigma_W.cpu().numpy()

def compute_Sigma_B(mu_c_dict, mu_G, _previous_classes):
    Sigma_B = 0
    K = len(mu_c_dict) if _previous_classes == 0 else len(mu_c_dict) + _previous_classes
    for i in range(_previous_classes, K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')
    device = WWT.device
    sub = (torch.eye(K, device=device) - 1 / K * torch.ones((K, K), device=device)) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()

def compute_W_H_relation(W, mu_c_dict, mu_G, _previous_classes):
    K = len(mu_c_dict) if _previous_classes == 0 else len(mu_c_dict) + _previous_classes
    H = torch.empty(mu_c_dict[_previous_classes].shape[0], K)
    for i in range(_previous_classes, K):
        H[:, i] = mu_c_dict[i] - mu_G

    WH = torch.mm(W, H.cuda(device=W.device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K, device=W.device) - 1 / K * torch.ones((K, K), device=W.device))

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H

def compute_Wh_b_relation(W, mu_G, b=None):
    Wh = torch.mv(W, mu_G.cuda(device=W.device))
    # res_b = torch.norm(Wh + b, p='fro')
    res_b = torch.norm(Wh, p='fro')
    return res_b.detach().cpu().numpy().item()


if __name__ == '__main__':  
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    init_cls = 0 if args['init_cls'] == args['increment'] else args['init_cls']
    PATH_TO_INFO = os.path.join(HARD_DRIVE_PATH, f'logs/{args["model_name"]}/{args["dataset"]}_init={init_cls}_incr={args["increment"]}/') 

    gamma_list = copy.deepcopy(args["gamma"])
    kd_gamma = args.get("kd_gamma", None)
    kd_gamma_list =  copy.deepcopy(kd_gamma) if kd_gamma is not None else [None]
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
 
    params = []
    for g in gamma_list:
        for kg in kd_gamma_list:
            p = os.path.join(PATH_TO_INFO,f"gamma={str(g).replace('.','_')}_kd_gamma={str(kg).replace('.','_')}_seed=")
            seeds = []
            for s in seed_list:
                _p = p + f'{s}/{INFO_FILE_NAME}'

                if os.path.exists(_p) :
                    seeds.append(s)
            if len(seeds) > 0 : 
                params.append({'g':g, 'kg':kg, 'seeds':seeds})
    
    for param in params : 
        args["gamma"] = param['g']
        args["kd_gamma"] = param['kg']
        for seed in param['seeds']:
            args["seed"] = seed
            args["device"] = device
            # RUN MAIN 
            main(args, main_path="logs/{}/{}_init={}_incr={}".format(args["model_name"],args["dataset"], init_cls, args['increment']))