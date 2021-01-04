import os
from os import path
import numpy as np
import sys
sys.path+=(r"utils", "data", "model")
import vis
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch

def train_val_dataset_split(dataset=None, vali_ratio=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=vali_ratio)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def fix_random_seed(seed_int=0):
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed(seed_int)
    np.random.seed(seed_int)
    torch.backends.cudnn.deterministic = True

def save_weight_info(wt_dict, save_dir=os.path.join('.', 'weights')):
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle
    save_fname = wt_dict["type"]+"_"+wt_dict["time"]+".pickle"
    with open(os.path.join(save_dir, save_fname), 'wb') as fp:
        pickle.dump(wt_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_weight_pickle(pickle_path):

    import pickle
    with open(pickle_path, 'rb') as pickle_file:
        wt_dict= pickle.load(pickle_file)
    return wt_dict

def model_select_cVAE_baseline(pkl_dir=None, sub_id=[], loss_function="MSE"):
    import glob
    pkl_path_all = glob.glob(os.path.join(pkl_dir,'*.pickle'))
    best_loss_sum = 1000000

    for pkl_path in pkl_path_all:
        wt_dict_candidate = load_weight_pickle(pickle_path=pkl_path)
        best_epoch = wt_dict_candidate["best epochs"]
        if wt_dict_candidate["type"][0:4] == "cGAN":
            continue
        if len(sub_id) != 0:
            if wt_dict_candidate["subjects included"] != sub_id:
                continue
        if wt_dict_candidate["loss function"] != loss_function:
                continue
        # get mean of loss for each epoch
        best_train_loss = np.mean(wt_dict_candidate["train loss list"][best_epoch])
        best_vali_loss = np.mean(wt_dict_candidate["validation loss list"][best_epoch])
        # select the bst one
        if best_loss_sum >  best_train_loss+best_vali_loss:
            best_loss_sum = best_train_loss + best_vali_loss
            best_wt_dict = wt_dict_candidate
    print(best_wt_dict["type"],best_wt_dict["loss function"])
    print("best loss sum: "+str(best_loss_sum))
    print("hidden layer size: "+str(best_wt_dict["hidden layer size"]))
    print("subjects included: " + str(best_wt_dict["subjects included"]))
    if best_wt_dict["type"][0:4] == "cVAE":
        print("code dimension: " + str(best_wt_dict["code dimension"]))
    print("learning rate decay speed: " + str(best_wt_dict["learning rate decay speed"]))
    print( "start learning rate: " + str(best_wt_dict[ "start learning rate"]))
    print( "learning rate update inter-times: " + str(best_wt_dict[ "learning rate update inter-times"]))
    print("leaky relu negative slope: " + str(best_wt_dict["leaky relu negative slope"]))
    print("maximum epochs: " + str(best_wt_dict["maximum epochs"]))
    print( "weight save path: " + best_wt_dict["weight save path"])
    print("\n")

    return best_wt_dict

def model_select_cGAN(pkl_dir=None, sub_id=[]):
    import glob
    pkl_path_all = glob.glob(os.path.join(pkl_dir, '*.pickle'))
    best_loss_sum = 1000000

    for pkl_path in pkl_path_all:
        wt_dict_candidate = load_weight_pickle(pickle_path=pkl_path)
        if wt_dict_candidate["type"][0:4] != "cGAN":
            continue
        if len(sub_id) != 0:
            if wt_dict_candidate["subjects included"] != sub_id:
                continue
        best_epoch = wt_dict_candidate["best epochs"]
        # get mean of loss for each epoch
        best_train_loss = np.mean(wt_dict_candidate["D_train loss list"][best_epoch])+\
                          np.mean(wt_dict_candidate["G_train loss list"][best_epoch])
        best_vali_loss =np.mean(wt_dict_candidate["D_validation loss list"][best_epoch])+\
                        np.mean(wt_dict_candidate["G_validation loss list"][best_epoch])
        # select the bst one
        if best_loss_sum > best_train_loss + best_vali_loss:
            best_loss_sum = best_train_loss + best_vali_loss
            best_wt_dict = wt_dict_candidate

    print(best_wt_dict["type"])
    print("best loss sum: " + str(best_loss_sum))
    print("hidden layer size: " + str(best_wt_dict["hidden layer size"]))
    print("subjects included: " + str(best_wt_dict["subjects included"]))
    print("code dimension: " + str(best_wt_dict["code dimension"]))
    print("learning rate decay speed: " + str(best_wt_dict["learning rate decay speed"]))
    print("start learning rate: " + str(best_wt_dict["start learning rate"]))
    print("learning rate update inter-times: " + str(best_wt_dict["learning rate update inter-times"]))
    print("leaky relu negative slope: " + str(best_wt_dict["leaky relu negative slope"]))
    print("maximum epochs: " + str(best_wt_dict["maximum epochs"]))
    print("G_weight save path: "+best_wt_dict["G_weight save path"])
    print("\n")


    return best_wt_dict

def plot_loss_from_dict(wt_dict=None):
    import matplotlib.pyplot as plt
    ## plot loss
    train_loss_list = np.array(wt_dict["train loss list"])
    vali_loss_list = np.array(wt_dict["validation loss list"])
    # for i in [10]:
    #     plt.plot(range(len(train_loss_list[i])), train_loss_list[i])
    plt.plot(range(wt_dict["maximum epochs"]), np.mean(train_loss_list, axis=1))
    plt.plot(range(wt_dict["maximum epochs"]), np.mean(vali_loss_list, axis=1))
    plt.ylim(0, 0.01)
    plt.show()

def load_best_model(type=None, sub_no=4, pkl_dir=os.path.join('.','results','selected_model'),
                    loss_function="MSE"):
    import glob
    wt_dict=[]
    pkl_path_all = glob.glob(os.path.join(pkl_dir, '*.pickle'))

    if type=="baseline":
        dip = 4
    else:
        dip =0

    for pkl_path in pkl_path_all:
        temp = os.path.split(pkl_path)
        pkl_name = temp[-1]
        if type =="cGAN":
            if pkl_name[:4] == type[:4] and pkl_name[13+dip]==str(sub_no):
                # print(pkl_path)
                wt_dict=load_weight_pickle(pkl_path)
        else:
            if pkl_name[:4] == type[:4] and pkl_name[13+dip]==str(sub_no) and pkl_name[15+dip:18+dip]==loss_function[:3]:

                wt_dict=load_weight_pickle(pkl_path)
    if len(wt_dict)>0:
        return wt_dict
    else:
        print("not exist")

def print_hyper_parameters():
    wt_dir = os.path.join('..', 'results', 'selected_model')
    import glob
    pkl_path_all = glob.glob(os.path.join(wt_dir, '*.pickle'))

    for pkl_path in pkl_path_all:

        wt_dict = load_weight_pickle(pickle_path=pkl_path)
        if wt_dict["type"][:4] == "cGAN":
            print(wt_dict["type"], wt_dict["hidden layer size"], wt_dict["subjects included"],
                  wt_dict["code dimension"], wt_dict["leaky relu negative slope"], wt_dict["train data size"])
        if wt_dict["type"][:4] == "base":
            print(wt_dict["type"], wt_dict["loss function"], wt_dict["hidden layer size"], wt_dict["subjects included"],
                  wt_dict["leaky relu negative slope"], wt_dict["train data size"])
        if wt_dict["type"][:4] == "cVAE":
            print(wt_dict["type"], wt_dict["loss function"], wt_dict["hidden layer size"], wt_dict["subjects included"],
                  wt_dict["code dimension"], wt_dict["leaky relu negative slope"], wt_dict["train data size"])


if __name__ == "__main__":
    print_hyper_parameters()