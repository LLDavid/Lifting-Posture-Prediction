import numpy as np
import os
import sys
sys.path+=("utils", "model")
from utils.utils import *
from shutil import copy
from model import models
import argparse

def model_test(type="baseline", HHW=[1.67, 0.6,0.6,0.45,0.45]):

    # pkl path
    wt_pkl_path = os.path.join(".","weights",type+".pickle")
    wt_dict = load_weight_pickle(pickle_path=wt_pkl_path)
    print(wt_dict["type"])

    # load to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # visualizer init
    pvis = vis.pose_visualizer()

    if type == "baseline":
        # load model from dict
        model = models.NN_baseline(nlayers=int(wt_dict["type"][-7]),
                                   whh=wt_dict["include subject height (whh)?"],
                                   in_size=wt_dict["hidden layer size"],
                                   lkns=wt_dict["leaky relu negative slope"]).to(device)
        model.load_state_dict(torch.load(wt_dict["weight save path"]))

        # posture prediction
        HHW = np.array(HHW).reshape((5,))
        y_pred = model(torch.tensor(HHW).float().to(device))
        y_pred = y_pred.detach().cpu().numpy()
        print("Predicted Posture: ", y_pred)

        # visualize
        pvis.pose37_3d(y_pred.reshape((105,)))

    if type == "cVAE":
        # load model from dict
        model = models.cVAE(nlayers=int(wt_dict["type"][-7]),
                            whh=wt_dict["include subject height (whh)?"],
                            in_size=wt_dict["hidden layer size"],
                            lkns=wt_dict["leaky relu negative slope"],
                            code_dim=wt_dict["code dimension"]).to(device)
        model.load_state_dict(torch.load(wt_dict["weight save path"]))

        # generate random code
        z_rand =  torch.randn(1, wt_dict["code dimension"], device=device)

        # posture prediction
        HHW = np.array(HHW).reshape((1,5))
        y_pred = model.decoder(z_rand.float(), torch.tensor(HHW).float().to(device))
        y_pred = y_pred.detach().cpu().numpy()

        # visualize
        pvis.pose37_3d(y_pred.reshape((105,)))

    if type == "cGAN":
        # load model from dict
        model = models.cGAN_G(nlayers=int(wt_dict["type"][-7]),
                              whh=wt_dict["include subject height (whh)?"],
                              in_size=wt_dict["hidden layer size"],
                              lkns=wt_dict["leaky relu negative slope"],
                              code_dim=wt_dict["code dimension"]).to(device)
        model.load_state_dict(torch.load(wt_dict["G_weight save path"]))

        # generate random code
        z_rand =  torch.randn(1, wt_dict["code dimension"], device=device)

        # posture prediction
        HHW = np.array(HHW).reshape((1,5))
        y_pred = model(z_rand.float(), torch.tensor(HHW).float().to(device))
        y_pred = y_pred.detach().cpu().numpy()

        # visualize
        pvis.pose37_3d(y_pred.reshape((105,)))

if __name__ == '__main__':
    # argumetn parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="type of model: baseline, cVAE, cGAN", type=str)
    parser.add_argument("--HHW", help="list of 5 numbers", type=float, nargs='+')
    args = parser.parse_args()

    if args.type and args.HHW: 
        model_test(type=args.type, HHW=args.HHW)
    else:
        model_test(type="baseline", HHW=[1.67, 0.6,0.6,0.45,0.45])

