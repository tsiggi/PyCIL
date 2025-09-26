import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

HARD_DRIVE_PATH = '/media/data/ctsiggiropoulos' 
check_average_accuracies = True
ONLY_FOR_NEW_TASKS = False
prefix = 'new_' if ONLY_FOR_NEW_TASKS else ''
title_sufix = ' [on the new classes of that task]'
INFO_FILE_NAME = prefix+'info.pkl'

models = ['lwf']
# gamma = [0, 0.05, 0.1, 0.2, 0.5]
# kd_gamma = [0, 0.05, 0.1, 0.2, 0.5]
gamma = [0, 0.05, 0.1, 0.2, 0.5]
kd_gamma = [0, 0.05, 0.1, 0.2, 0.5]
seed = [0, 1, 2]
colors = ['#1f77b4' , "#ff7f0e" , "#2ca02c" ,'#d62728' , "#9467bd" , "#8c564b"]
# colors = ["r", "g", "b", "c", "m", "y"]   # # red, green, blue, cyan, magenta, yellow
markers = ["o", "s", "v", "^", "D", "x"]  # circle, square, down-triangle, up-triangle, diamond, "x" 

datasets = ['mnist', 'cifar10', 'cifar100']

# ------------------------ plot for figure 3 mnist ---------------------------------------------------------------------
# id = 0
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '/'
# PATH_TO_INFO_SGD = os.path.join(PATH_TO_INFO, 'SGD_info_new.pkl')
# PATH_TO_INFO_ADAM = os.path.join(PATH_TO_INFO, 'Adam_info_new.pkl')
# PATH_TO_INFO_LBFGS = os.path.join(PATH_TO_INFO, 'LBFGS_info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_SGD, 'rb') as f:
#     info_sgd = pickle.load(f)
#
# with open(PATH_TO_INFO_ADAM, 'rb') as f:
#     info_adam = pickle.load(f)
#
# with open(PATH_TO_INFO_LBFGS, 'rb') as f:
#     info_lbfgs = pickle.load(f)

# ------------------------ plot for figure 3 cifar 10 ------------------------------------------------------------------
# id = 1
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '/'
# PATH_TO_INFO_SGD = os.path.join(PATH_TO_INFO, 'SGD_info_new.pkl')
# PATH_TO_INFO_ADAM = os.path.join(PATH_TO_INFO, 'Adam_info_new.pkl')
# PATH_TO_INFO_LBFGS = os.path.join(PATH_TO_INFO, 'LBFGS_info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_SGD, 'rb') as f:
#     info_sgd = pickle.load(f)
#
# with open(PATH_TO_INFO_ADAM, 'rb') as f:
#     info_adam = pickle.load(f)
#
# with open(PATH_TO_INFO_LBFGS, 'rb') as f:
#     info_lbfgs = pickle.load(f)

# ------------------------ plot for figure 6 mnist ---------------------------------------------------------------------
# id = 0
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '_sota/'
# PATH_TO_INFO_ETFfc_false_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'false_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_true_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'false_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_false_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'true_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_true_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'true_'+'info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_ETFfc_false_fixdim_false, 'rb') as f:
#     info_ETFfc_false_fixdim_false = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_true_fixdim_false, 'rb') as f:
#     info_ETFfc_true_fixdim_false = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_false_fixdim_true, 'rb') as f:
#     info_ETFfc_false_fixdim_true = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_true_fixdim_true, 'rb') as f:
#     info_ETFfc_true_fixdim_true = pickle.load(f)

# ------------------------ plot for figure 6 cifar 10 ------------------------------------------------------------------
id = 2
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '_sota/'
PATH_TO_INFO = os.path.join(HARD_DRIVE_PATH, f'logs/{models[0]}/{datasets[id]}_init=0_incr=10/') 

paths = []
for g in gamma:
    for kg in kd_gamma:
        p = os.path.join(PATH_TO_INFO,f"gamma={str(g).replace('.','_')}_kd_gamma={str(kg).replace('.','_')}_seed=")
        seed_list = []
        for s in seed:
            _p = p + f'{s}/{INFO_FILE_NAME}'
            if os.path.exists(_p) :
                seed_list.append(s)
        if len(seed_list) > 0 : 
            paths.append({'path':p, 'seeds':seed_list, 'g':g, 'kg':kg})

# print(paths)

# LOAD INFO DATA
for i, path in enumerate(paths): 
    model_data = []
    for s in path['seeds']: 
        with open(path['path']+f"{s}/{INFO_FILE_NAME}", 'rb') as f:
            model_data.append(pickle.load(f))
    paths[i]['data'] = model_data


# XTICKS = [0, 50, 100, 150, 200]

def plot_collapse(test=False):
    fig = plt.figure(figsize=(10, 8))
    test_sufix = '_test' if test else '' 
    test_prefix = 'test_' if test else '' 

        'collapse_metric_test': [],
        'WH_relation_metric_test': [],
        'Wh_b_relation_metric_test': [],

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['collapse_metric'+test_sufix], color= colors[i%6], label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.title('Collapse_metric{} on {}'.format(test_sufix, paths[j]['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.legend(fontsize=20)
        plt.grid(True)
        
        if not os.path.exists(path['path'][:-6]):
            os.makedirs(path['path'][:-6])
        fig.savefig(path['path'][:-6]+ f"/{prefix}{test_prefix}NC1.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_collapse()] saved on: " + path['path'][:-6] + "/NC1.png")
    plt.close()

    # plt.plot(info_ETFfc_false_fixdim_false['collapse_metric'], 'c', marker='v', ms=16, markevery=15, linewidth=5, alpha=0.7)
    # plt.plot(info_ETFfc_true_fixdim_false['collapse_metric'], 'b', marker='o', ms=16, markevery=15, linewidth=5, alpha=0.7)
    # plt.plot(info_ETFfc_false_fixdim_true['collapse_metric'], 'g', marker='s', ms=16, markevery=15, linewidth=5, alpha=0.7)
    # plt.plot(info_ETFfc_true_fixdim_true['collapse_metric'], 'r', marker='X', ms=16, markevery=15, linewidth=5, alpha=0.7)

    # plt.xlabel('Epoch', fontsize=40)
    # plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    # plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(-0.1, 0.41, 0.1), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(-2, 6.01, 2), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(-0.1, 0.21, 0.1), fontsize=30) # plot for figure 6 mnist
    # plt.yticks(np.arange(0, 12.1, 4), fontsize=30) # plot for figure 6 cifar 10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    # plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, -0.01, 0.4]) # plot for figure 3 mnist
    # plt.axis([0, 200, -0.2, 6]) # plot for figure 3 cifar10
    # plt.axis([0, 200, -0.01, 0.2]) # plot for figure 6 mnist
    # plt.axis([0, 200, -0.4, 12]) # plot for figure 6 cifar 10

    # fig.savefig(out_path + datasets[id] + "-resnet18-NC1.pdf", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))

    for j, path in enumerate(paths): 
        plt.clf()
        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['ETF_metric'], color= colors[i%6], label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('ETF_metric on {}'.format(path['path'][33:-6].replace('/',' ').replace('_',' ') + title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}NC2.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_ETF()] saved on : " + path['path'][:-6] + "/NC2.png")
    plt.close()


def plot_WH_relation(test=False):
    fig = plt.figure(figsize=(10, 8))
    test_sufix = '_test' if test else '' 
    test_prefix = 'test_' if test else '' 
    
    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['WH_relation_metric'+test_sufix], color= colors[i%6], label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('WH_relation_metric{} on {}'.format(test_sufix, path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}{test_prefix}NC3.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_WH_relation()] saved on : " + path['path'][:-6] + "/NC3.png")
    plt.close()


def plot_residual(test=False):
    fig = plt.figure(figsize=(10, 8))
    test_sufix = '_test' if test else '' 
    test_prefix = 'test_' if test else '' 

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['Wh_b_relation_metric'+test_sufix], color= colors[i%6], label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('Wh_b_relation_metric{} on {}'.format(test_sufix, path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}{test_prefix}NC4.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_WH_relation()] saved on : " + path['path'][:-6] + "/NC3.png")
    plt.close()
    

def plot_train_test_acc():
    fig = plt.figure(figsize=(10, 8))

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['train_acc1'], color= colors[0], label=f"train, seed={path['seeds'][i]}", marker=markers[(2*i)%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
            plt.plot(np.arange(1, 10.1), info['test_acc1'], color= colors[3], label=f"test, seed={path['seeds'][i]}", marker=markers[(2*i+1)%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('Train & Test accuracy on {}'.format(path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel('Accuracy', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}train-test-acc.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_train_acc()] saved on : " + path['path'][:-6] + "/NC3.png")
    plt.close()


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['train_acc1'], color= colors[i%6],label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('Train accuracy on {}'.format(path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel('Train accuracy', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}train-acc.png", bbox_inches='tight')
        # print(">>> [INFO] [plot_train_acc()] saved on : " + path['path'][:-6] + "/NC3.png")
    plt.close()

def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['test_acc1'], color= colors[i%6], label=f"seed={path['seeds'][i]}", marker=markers[i%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title('Test accuracy on {}'.format(path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel('Test accuracy', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}test-acc.png", bbox_inches='tight')
       # print(">>> [INFO] [plot_test_acc()] saved on : " + PATHS_TO_INFO[i][:-8]+ "test-acc.png")
    plt.close()

def plot_sigmas(): 
    fig = plt.figure(figsize=(10, 8))

    for j, path in enumerate(paths): 
        plt.clf()

        for i, info in enumerate(path['data']):
            plt.plot(np.arange(1, 10.1), info['trace_sigma_w'], color= colors[4], label=f"tr(sigma_w), seed={path['seeds'][i]}", marker=markers[(2*i)%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
            plt.plot(np.arange(1, 10.1), info['trace_sigma_b'], color= colors[2], label=f"tr(sigma_b), seed={path['seeds'][i]}", marker=markers[(2*i+1)%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

        plt.title("Trace's of Sigma's on {}".format(path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
        plt.xlabel('Tasks', fontsize=40)
        plt.ylabel('Traces of Sigma', fontsize=40)
        plt.xticks(np.arange(1, 10.1), fontsize=30)
        plt.figtext(0.5, -0.05, "High tr(s_w)-> NC1 incr & High tr(s_b) -> NC1 dec.", ha="center", fontsize=22)
        plt.legend(fontsize=20)
        plt.grid(True)

        fig.savefig(path['path'][:-6]+ f"/{prefix}sigma-traces.png", bbox_inches='tight')
       # print(">>> [INFO] [plot_test_acc()] saved on : " + PATHS_TO_INFO[i][:-8]+ "test-acc.png")
    plt.close()


# def plot_sigmas_ratio(): 
#     fig = plt.figure(figsize=(10, 8))

#     for j, path in enumerate(paths): 
#         plt.clf()

#         for i, info in enumerate(path['data']):
#             plt.plot(np.arange(1, 10.1), info['trace_sigma_w']/info['trace_sigma_b'], color= colors[(i+3)%6], label=f"tr(sigma_w)/tr(sigma_b), seed={path['seeds'][i]}", marker=markers[(2*i)%6], ms=16, markevery=1, linewidth=5, alpha=0.7)

#         plt.title(". {}".format(path['path'][33:-6].replace('/',' ').replace('_',' ')+ title_sufix))
#         plt.figtext(0.5, -0.05, "high tr(s_w)-> NC1 incr, high tr(s_b) -> NC1 dec.", ha="center", fontsize=12)
#         plt.xlabel('Tasks', fontsize=40)
#         plt.ylabel('Traces of Sigma', fontsize=40)
#         plt.xticks(np.arange(1, 10.1), fontsize=30)
#         plt.legend(fontsize=20)
#         plt.grid(True)

#         fig.savefig(path['path'][:-6]+ f"/{prefix}sigma-traces.png", bbox_inches='tight')
#        # print(">>> [INFO] [plot_test_acc()] saved on : " + PATHS_TO_INFO[i][:-8]+ "test-acc.png")
#     plt.close()

def plot_average_accuracies(): 
    fig = plt.figure(figsize=(20, 15))
    cnt = 0 
    for j, path in enumerate(paths): 
        if path['kg'] != 0 : 
            continue

        average_test_acc = np.mean([path['data'][i]['test_acc1'] for i in range(len(path['data']))], axis=0)  
        average_train_acc = np.mean([path['data'][i]['train_acc1'] for i in range(len(path['data']))], axis=0)  

        plt.plot(np.arange(1, 10.1), average_train_acc, color= colors[cnt%6], label=f"train acc g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
        plt.plot(np.arange(1, 10.1), average_test_acc, color= colors[cnt%6], label=f"test acc g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
        cnt += 1

    plt.title("Average Accuracies for different seeds on {}".format(models[0]))
    plt.xlabel('Tasks', fontsize=40)
    plt.ylabel('Accuracies', fontsize=40)
    plt.xticks(np.arange(1, 10.1), fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)

    fig.savefig(PATH_TO_INFO+ f"{prefix}average_accuracies.png", bbox_inches='tight')
    # print(">>> [INFO] [plot_test_acc()] saved on : " + PATHS_TO_INFO[i][:-8]+ "test-acc.png")
    plt.close()

def plot_average_accuracy_improvements(print_std=False): 
    fig = plt.figure(figsize=(20, 15))
    cnt = 0 
    base_ave_test_acc, base_ave_train_acc = None, None
    for j, path in enumerate(paths): 
        if path['g'] == 0 and path['kg'] == 0: 
            base_train_acc = np.array([path['data'][i]['train_acc1'] for i in range(len(path['data']))]) 
            base_test_acc = np.array([path['data'][i]['test_acc1'] for i in range(len(path['data']))])
    
    if base_test_acc is None : 
        return 

    for j, path in enumerate(paths):
        if path['kg'] != 0 or (path['g'] == 0 and path['kg'] == 0):  
            continue
        
        _dif_from_base_train = np.array([path['data'][i]['train_acc1'] for i in range(len(path['data']))]) - base_train_acc 
        _dif_from_base_test = np.array([path['data'][i]['test_acc1'] for i in range(len(path['data']))]) - base_test_acc 

        av_incr_train = np.mean(_dif_from_base_train, axis=0)
        av_incr_test = np.mean(_dif_from_base_test, axis=0)
        std_incr_train = np.std(_dif_from_base_train, axis=0)
        std_incr_test = np.std(_dif_from_base_test, axis=0)

        if print_std : 
            plt.errorbar(np.arange(1, 10.1), av_incr_train, yerr=std_incr_train, fmt='-' , color= colors[cnt%6], label=f"train g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
            plt.errorbar(np.arange(1, 10.1), av_incr_test, yerr=std_incr_test, fmt='--' , color= colors[cnt%6], label=f"test g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
        else : 
            plt.plot(np.arange(1, 10.1), av_incr_train, color= colors[cnt%6], label=f"train acc g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
            plt.plot(np.arange(1, 10.1), av_incr_test, color= colors[cnt%6], label=f"test acc g:{path['g']}", marker=markers[cnt%6], ms=16, markevery=1, linewidth=5, alpha=0.7)
        cnt += 1

    plt.title("{} average accuracy improvements for different g values".format(models[0]))
    plt.xlabel('Tasks', fontsize=40)
    plt.ylabel('Accuracy (%)', fontsize=40)
    plt.xticks(np.arange(1, 10.1), fontsize=30)
    plt.legend(fontsize=20)
    plt.grid(True)

    fig.savefig(PATH_TO_INFO+ f"{prefix}average_accuracies_improvements.png", bbox_inches='tight')
    # print(">>> [INFO] [plot_test_acc()] saved on : " + PATHS_TO_INFO[i][:-8]+ "test-acc.png")
    plt.close()



def main():

    if check_average_accuracies : 
        plot_average_accuracy_improvements()
        # plot_average_accuracies()
        exit()

    plot_collapse()
    plot_ETF()
    plot_WH_relation()
    plot_residual()

    plot_train_test_acc()
    # plot_train_acc()
    # plot_test_acc()

    if not ONLY_FOR_NEW_TASKS: 
        plot_sigmas()
    

if __name__ == "__main__":
    main()