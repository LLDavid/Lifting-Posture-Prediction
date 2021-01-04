import torch
import sys
import matplotlib.pyplot as plt
sys.path+=("..", "utils", "data", "model")
# from data import datasets
# from model import models
# from utils import vis
# from utils import metrics
from utils import *
import scipy.io as sio
from torch.utils.data import DataLoader
import os
import numpy as np
import vis


def get_2D_proj():
    pose_all=np.load(r'F:\Research\MyPaper\IKGan\pycode\data\MOPED25\x_moped25_lift.npy')
    pvis = vis.pose_visualizer()
    plt_h = pvis.image_2D(pose_all=pose_all[5])
def get_length_angle_distribution():
    # load raw data
    x = np.load(os.path.join('..','data','MOPED25','x_moped25.npy'))
    y = np.load(os.path.join('..','data','MOPED25','y_moped25.npy'))
    print(x.shape, y.shape)
    # metrics class
    pose_metrics = metrics.gp_metrics()
    length =[]
    angle =[]
    for i in range(len(y)):
        if np.count_nonzero(np.isnan(x[i,:]))==0: # select tpose and exclude nan values
            angle.append(pose_metrics.seg_angles(pose=x[i,:]))
            length.append(pose_metrics.seg_lengths(pose=x[i,:]))
        else:
            continue
    # convert to numpy
    angle = np.array(angle).reshape((len(angle), -1))
    length = np.array(length).reshape((len(length), -1))
    print(angle.shape, length.shape)
    # save path
    angle_save_path = os.path.join('..','results','pose_metrics','angle_all.npy')
    length_save_path = os.path.join('..', 'results', 'pose_metrics', 'length_all.npy')
    # save
    np.save(angle_save_path,  angle)
    np.save(length_save_path, length)
def save_subject_length_ratio():
    length_all = np.load(os.path.join('..', 'results', 'pose_metrics', 'length_all.npy'))
    height_all = np.load(os.path.join('..', 'data', 'MOPED25', 'sub_height.npy'))
    y_all = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))


    length_ratio_all = length_all
    for i in range(len(length_ratio_all)):
        length_ratio_all[i, :] /= height_all[int(y_all[i, 0]-1)]*10
    np.save(r"F:\Research\MyPaper\IKGan\pycode\results\pose_metrics\length_ratio_all.npy", length_ratio_all)

def tpose_length_angle_hist():
    tpose_angle = np.load(os.path.join('..','results','pose_metrics','tpose_angle.npy'))
    tpose_length = np.load(os.path.join('..', 'results', 'pose_metrics', 'tpose_length.npy'))
    lift_angle = np.load(os.path.join('..', 'results', 'pose_metrics', 'lift_angle.npy'))
    lift_length = np.load(os.path.join('..', 'results', 'pose_metrics', 'lift_length.npy'))
    plt.interactive(False)
    _ = plt.hist(tpose_angle[:,0], bins='auto', color='r', alpha=0.5)
    _ = plt.hist(lift_angle[:, 0], bins='auto', color='b', alpha=0.5)
    plt.show()
def tpose_angle_hist_by_subject():
    angle_all = np.load(os.path.join('..','results','pose_metrics','angle_all.npy'))
    length_all = np.load(os.path.join('..', 'results', 'pose_metrics', 'length_all.npy'))
    y_all =  np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))
    print(angle_all.shape, length_all.shape, y_all.shape)
    figs, axs = plt.subplots(angle_all.shape[1],11, sharey=True)
    for i in range(2,13):
        for j in range(angle_all.shape[1]):
            angle_temp = angle_all[(y_all[:,0]==i)&(y_all[:,1]<=24)&(y_all[:,1]>=13),j]
            print(i,angle_temp.shape)
            _ = axs[j, i-2].hist(angle_temp, bins='auto', color=np.random.rand(3), alpha=0.4)
    plt.show()
def tpose_length_hist_by_subject():
    length_all = np.load(os.path.join('..', 'results', 'pose_metrics', 'length_all.npy'))
    y_all = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))
    print(length_all.shape, y_all.shape)
    figs, axs = plt.subplots(length_all.shape[1], 11, sharey=True)
    for i in range(2, 13):
        for j in range(length_all.shape[1]):
            length_temp = length_all[(y_all[:, 0] == i) & (y_all[:, 1] <= 24) & (y_all[:, 1] >= 13), j]
            _ = axs[j, i - 2].hist(length_temp, bins='auto', color=np.random.rand(3), alpha=0.4)
    plt.show()
def tpose_length_ratio_hist_by_subject():
    length_all = np.load(os.path.join('..', 'results', 'pose_metrics', 'length_ratio_all.npy'))
    y_all = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))
    print(length_all.shape, y_all.shape)
    figs, axs = plt.subplots(length_all.shape[1], 11, sharey=True)
    # pesudo matrix for means
    mean_mat = np.ones((length_all.shape[1], 11))
    for i in range(2, 13):
        for j in range(length_all.shape[1]):
            length_temp = length_all[(y_all[:, 0] == i) & (y_all[:, 1] <= 24) & (y_all[:, 1] >= 13), j]
            _ = axs[j, i - 2].hist(length_temp, bins='auto', color=np.random.rand(3), alpha=0.4)
            mean_mat[j, i-2] = np.mean(length_temp)
    print(mean_mat)
    plt.show()

def WH_hist():
    WH = np.load(
        r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WH_gcs.npy")
    plt.interactive(False)
    # _ = plt.hist(WH[:,0] / 1000.0, bins='auto', alpha=0.5)
    _ = plt.hist(WH[:, 1] / 1000.0, bins='auto', alpha=0.5)
    plt.show()

def get_file_name(sub_id, task_id, task_iid, isbox=None):
    if isbox:
        file_name = 'sub' + "{0:0=2d}".format(sub_id) + '_' + "{0:0=2d}".format(task_id) + '_' + "{0:0=2d}".format(task_iid)+'_box_trc'
    else:
        file_name='sub'+"{0:0=2d}".format(sub_id)+'_'+"{0:0=2d}".format(task_id)+'_'+"{0:0=2d}".format(task_iid)+'_trc'
    return file_name

def save_subject_height():
    height = np.array([179, 188, 178, 167, 167, 185, 180, 183, 165, 167, 167, 183]).reshape((12,1))
    print(height.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\sub_height.npy", height)


def save_pose_mat_to_npy():
    import scipy.io as sio
    file_dir=r"F:\Data\working_posture_trc_mat"
    file_dir_box = r"F:\Data\box_trc_mat"
    # pose data and label
    pure_pose_37=np.zeros((1,111))
    pose_label=np.zeros((1,4))
    box_10 = np.zeros((1,30))
    file_cnt=[0,0]
    for i in range(2,13):
        print("The "+str(i)+" th subject")
        for j in range(1,26):
            for k in range(1,5):
                # pose path
                file_path_0=os.path.join(file_dir, 'sub'+"{0:0=2d}".format(i))
                file_name=get_file_name(i,j,k)
                file_path_1=os.path.join(file_path_0, file_name+'.mat')
                # box path
                file_name_box = get_file_name(i, j, k, isbox=True)
                file_path_0_box = os.path.join(file_dir_box, 'sub' + "{0:0=2d}".format(i))
                file_path_1_box = os.path.join(file_path_0_box, file_name_box + '.mat')
                if path.exists(file_path_1) and path.exists(file_path_1_box): # check file if exists
                    file_cnt[0]+=1
                    # print(file_path_1)
                    try: # some with NaN values cannot be loaded
                        pose_all = sio.loadmat(file_path_1)
                        box_all =  sio.loadmat(file_path_1_box)
                    except:
                        continue
                        file_cnt[1] += 1
                    # create framewise label
                    pose_id = pose_all['xyz_all'][0][0]
                    label_temp = np.zeros((len(pose_id), 3))
                    label_temp = np.hstack((label_temp,pose_id))
                    # pose_time = pose_all['xyz_all'][1]
                    #label_temp =
                    for ii in range(37):
                        joint_xyz=pose_all['xyz_all'][2][0][ii][1]
                        # if type(joint_xyz).__module__ != np.__name__:
                        #     print(i,j,k,ii)
                        #     print(joint_xyz)
                        #     exit()
                        if ii==0:
                            pose_temp=joint_xyz
                        else:
                            pose_temp=np.hstack((pose_temp, joint_xyz))
                    for ii in range(10):
                        box_joint_xyz = box_all['box_xyz_all'][2][0][ii][1]
                        if ii == 0:
                            box_temp = box_joint_xyz
                        else:
                            box_temp = np.hstack((box_temp, box_joint_xyz))
                    try:
                        pure_pose_37=np.vstack((pure_pose_37, pose_temp))
                        label_temp[:, 0:3]=[i,j,k]
                        pose_label=np.vstack((pose_label, label_temp))
                        box_10 = np.vstack(( box_10,  box_temp))
                    except:
                        print(i,j,k,ii)
                        exit()
    # delete 1 st zeros row
    pure_pose_37=np.delete(pure_pose_37, 0, 0)
    box_10 = np.delete(box_10 , 0, 0)
    # pure_pose_37[np.isnan(pure_pose_37)]=-1
    pure_pose_37.astype('float')
    pose_label=np.delete(pose_label,0,0)
    pose_label.astype('float')
    print(pure_pose_37.shape, pose_label.shape, box_10.shape)
    print(np.count_nonzero(np.isnan(pure_pose_37)), np.count_nonzero(np.isnan(box_10)))
    exit()
    # delete nan values
    nan_id_temp=[]
    for jj in range(len(pose_label)):
        if np.count_nonzero(np.isnan(pure_pose_37[jj,:]))>0: # delete the pose with nan
            nan_id_temp.append(jj)
    pure_pose_37 = np.delete(pure_pose_37, nan_id_temp, 0)
    pose_label = np.delete(pose_label, nan_id_temp, 0)
    print(pure_pose_37.shape, pose_label.shape)
    print(np.count_nonzero(np.isnan(pure_pose_37)))

    ## save gcs of pose37
    pose37_gcs_save_path = os.path.join('F:\Research\MyPaper\IKGan\pycode',
                                    'data', 'MOPED25', 'x_moped25_gcs.npy')
    np.save(pose37_gcs_save_path, pure_pose_37)

    # normalize w.r.t. center of L/R A/PSIS
    # get the mean
    xc=np.mean(pure_pose_37[:, [3*(kk-1) for kk in [18, 19, 20, 21]]],-1, keepdims=True)
    yc=np.mean(pure_pose_37[:, [3*(kk-1)+1 for kk in [18, 19, 20, 21]]],-1, keepdims=True)
    zc=np.mean(pure_pose_37[:, [3*(kk-1)+2 for kk in [18, 19, 20, 21]]],-1, keepdims=True)
    print(xc.shape, yc.shape, zc.shape)
    pure_pose_37[:, [3*(kk-1) for kk in range(37)]] -= xc
    pure_pose_37[:, [3 * (kk - 1)+1 for kk in range(37)]] -= yc
    pure_pose_37[:, [3 * (kk - 1)+2 for kk in range(37)]] -= zc

    # save
    pose37_save_path=os.path.join('F:\Research\MyPaper\IKGan\pycode',
                           'data','MOPED25','x_moped25.npy')
    pose_label_save_path=os.path.join('F:\Research\MyPaper\IKGan\pycode',
                           'data','MOPED25','y_moped25.npy')

    np.save(pose37_save_path, pure_pose_37)
    np.save(pose_label_save_path, pose_label)
def save_box_mat_to_npy():
    import scipy.io as sio
    file_dir = r"F:\Data\box_trc_mat"
    # pose data and label
    box_10 = np.zeros((1, 30))
    box_label = np.zeros((1,4))
    file_cnt = 0
    for i in range(2, 13):
        print("The " + str(i) + " th subject")
        for j in range(13, 25):
            for k in range(1, 4):
                file_path_0 = os.path.join(file_dir, 'sub' + "{0:0=2d}".format(i))
                file_name = get_file_name(i, j, k, isbox=True)
                file_path_1 = os.path.join(file_path_0, file_name + '.mat')
                if path.exists(file_path_1):  # check file if exists
                    file_cnt += 1
                    try:  # some with NaN values cannot be loaded
                        box_all = sio.loadmat(file_path_1)
                    except:
                        continue
                        file_cnt += 1
                    # create framewise label
                    box_id = box_all['box_xyz_all'][0][0]
                    label_temp = np.zeros((len(box_id), 3))
                    label_temp = np.hstack((label_temp, box_id))
                    for ii in range(10):
                        joint_xyz = box_all['box_xyz_all'][2][0][ii][1]
                        if ii == 0:
                            box_temp = joint_xyz
                        else:
                            box_temp = np.hstack((box_temp, joint_xyz))
                    try:
                        box_10 = np.vstack((box_10, box_temp))
                        label_temp[:, 0:3] = [i, j, k]
                        box_label = np.vstack((box_label, label_temp))
                    except:
                        print(i, j, k, ii, "exit")
                        exit()
    # delete first row
    box_10 = np.delete(box_10, 0, 0)
    box_10.astype('float')
    box_label = np.delete(box_label, 0, 0)
    box_label.astype('float')
    print(box_10.shape,box_label.shape)
    print(np.count_nonzero(np.isnan(box_10)))
    # delete nan values
    nan_id_temp = []
    for jj in range(len(box_label)):
        if np.count_nonzero(np.isnan(box_10[jj, :])) > 0:  # delete the pose with nan
            nan_id_temp.append(jj)
    box_10 = np.delete(box_10, nan_id_temp, 0)
    box_label = np.delete(box_label, nan_id_temp, 0)
    print(box_10.shape, box_label.shape)
    print(np.count_nonzero(np.isnan(box_10)))

    ## save gcs of pose37
    box_gcs_save_path = os.path.join('..',
                                        'data', 'MOPED25', 'box_13-24_gcs.npy')
    box_label_path = os.path.join('..',
                                     'data', 'MOPED25', 'box_13-24_label.npy')
    np.save(box_gcs_save_path, box_10)
    np.save(box_label_path, box_label)

def save_chair_shelf_to_npy():
    #'V_front_left_6', 'V_behind_left_6','V_front_right_6', 'V_behind_right_6'
    #'V_front_left_5', 'V_behind_left_5','V_front_right_5', 'V_behind_right_5'
    #'V_front_left_4'    'V_behind_left_4'    'V_front_right_4'    'V_behind_right_4'
    #'V_front_left_3'    'V_behind_left_3'    'V_front_right_3'    'V_behind_right_3'
    #'V_front_left_2'    'V_behind_left_2'    'V_front_right_2'    'V_behind_right_2'
    #'V_front_left_1'    'V_behind_left_1'    'V_front_right_1'    'V_behind_right_1'
    #'V_behind_middle_6'    'V_right_middle_6'
    shelf_xyz = np.array([[1198, 1626, 206], [1287, 1623, 434], [617, 1632, 434], [705, 1627, 667],
                         [1190, 1294, 197], [1281, 1287, 434], [608, 1296, 430], [702, 1295, 668],
                         [1184, 980, 193], [1278, 983, 431], [604, 987, 428], [699, 985, 664],
                         [1179, 676, 190], [1273, 668, 426], [598, 675, 429], [693, 674, 666],
                         [1173, 361, 189], [1268, 357, 424], [592, 367, 427], [688, 362, 666],
                         [1168, 51, 187], [1263, 47, 424], [587, 56, 428], [684, 52, 661],
                         [1011, 1625, 545], [676, 1648, 544]])
    print(shelf_xyz.shape)
    #'V_back_up_right'    'V_back_middle_right'    'V_leg_back_right_up'    'V_leg_back_right_down'
    #'V_back_up_left'    'V_leg_back_left_up'    'V_leg_back_left_down'    'V_leg_front_right_up'
    #'V_leg_front_right_down'    'V_leg_front_left_up'    'V_leg_front_left_down'
    chair_xyz = np.array([[-190.287140000000,787.601200000000,-1226.18018000000],[-173.289200000000,626.272890000000,-1215.22998000000],
[-195.399990000000,416.750000000000,-1154.19995000000],[-204.270000000000,24.7400000000000,-1203.66003000000],
                          [212.657270000000,785,-1224.09998000000],[205.759990000000,425.429990000000,-1161.94995000000],
                          [191.480000000000,23.2300000000000,-1217.34998000000],[-206.770000000000,458.510010000000,-752.219970000000],
[-202.770000000000,22.6900000000000,-753.260010000000],[236.899990000000,448.140010000000,-769.989990000000],
                          [217.380000000000,22.6000000000000,-766.190000000000]])
    print(chair_xyz.shape)
    np.save("F:\Research\MyPaper\IKGan\pycode\data\MOPED25\shelf.npy", shelf_xyz)
    np.save("F:\Research\MyPaper\IKGan\pycode\data\MOPED25\chair.npy", chair_xyz)

def get_WH():
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift.npy'))
    y_data = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25_lift.npy'))

    # minimum of right big toe and Caudal Calcaneus
    floor_height = np.minimum(x_data[:, (32-1)*3+1], x_data[:, (34-1)*3+1])
    # middle of RS and us
    right_wrist_height = (x_data[:, (14-1)*3+1]+x_data[:, (16-1)*3+1]) / 2
    # target height
    target_height=  right_wrist_height-floor_height
    target_height=target_height.reshape((len(target_height),1))

    target_width = []

    for i in range(len(x_data)):
        # distance to plane formed by R\L CC and (0,1,0)
        up = np.array([0,1,0])
        # calculate the normal vector of the ref plane
        zn=1
        xn=-(x_data[i, 3*(34-1)+2]-x_data[i, 3*(35-1)+2])/(x_data[i, 3*(34-1)]-x_data[i, 3*(35-1)])
        nv=np.array([xn, 0, zn])
        nv /= np.linalg.norm(nv)
        # check nv direction
        v_34_32 = x_data[i, (32- 1) * 3 : (32 - 1) * 3 + 3] -\
                              x_data[i, (34 - 1) * 3 : (34 - 1) * 3 + 3]
        if np.dot(nv, v_34_32)<0:
            nv=-nv
        # middle of RS and us
        middle_wrist = (x_data[i, (14 - 1) * 3 : (14 - 1) * 3 + 3] +
                              x_data[i, (16 - 1) * 3 : (16 - 1) * 3 + 3])/2
        ref_v = middle_wrist - x_data[i, (34 - 1) * 3 : (34 - 1) * 3 + 3]
        # if np.dot(nv, ref_v)<-300:
        #     pvis = vis.pose_visualizer()
        #     pvis.pose37_3d(pose_all=x_data[i,:])

        # calculate the distance as projection of ref vector on normal direction
        target_width.append(np.dot(nv, ref_v))
    target_width=np.array(target_width).reshape((len(target_height),1))
    print(y_data.shape, target_height.shape, target_width.shape)
    lift_WH = np.hstack((target_width, target_height))
    print(lift_WH.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WH.npy",lift_WH)


    # import matplotlib.pyplot as plt
    # _ = plt.hist(target_width, bins='auto')
    # plt.show()

def save_lift_pose():
    # lcs and gcs
    x_data_gcs = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_gcs.npy'))
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25.npy'))

    y_data = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))

    lift_cnt=0
    for i in range(len(y_data)): # 1-25
        if y_data[i, 1] in range(13, 25):
            lift_cnt+=1
            if lift_cnt is 1:
                x_data_lift=x_data[i, :]
                x_data_lift_gcs=x_data_gcs[i, :]
                y_data_lift=y_data[i,:]
            else:
                x_data_lift=np.vstack((x_data_lift, x_data[i, :]))
                x_data_lift_gcs= np.vstack((x_data_lift_gcs, x_data_gcs[i, :]))
                y_data_lift = np.vstack((y_data_lift, y_data[i, :]))

    print(lift_cnt)
    print(x_data_lift_gcs.shape, x_data_lift.shape, y_data_lift.shape)

    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\x_moped25_lift.npy", x_data_lift)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\x_moped25_lift_gcs.npy", x_data_lift_gcs)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\y_moped25_lift.npy", y_data_lift)

def get_WH_gcs():
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift_gcs.npy'))
    shelf_xyz=np.load(os.path.join('..', 'data', 'MOPED25', 'shelf.npy'))

    target_height=[]
    target_width=[]
    for i in range(len(x_data)):
        # middle of RS and us as reference height
        target_height.append((x_data[i, (14 - 1) * 3+1] +
                        x_data[i, (16 - 1) * 3+1]) / 2)
        # use the distance difference between middle of R/L CC and wrist
        middle_cc = (x_data[i, (34- 1) * 3: (34- 1) * 3 + 3] +
                       x_data[i, (35 - 1) * 3: (35- 1) * 3 + 3]) / 2
        middle_wrist = (x_data[i, (14- 1) * 3: (14- 1) * 3 + 3] +
                       x_data[i, (16 - 1) * 3: (16- 1) * 3 + 3]) / 2

        nv = (shelf_xyz[22,:]-shelf_xyz[23,:]).astype(np.float)
        nv /= np.linalg.norm(nv) # unit normal vector


        ref_01=middle_wrist-shelf_xyz[22,:]
        ref_02 = middle_cc - shelf_xyz[22, :]

        # distance
        target_width.append(np.dot(ref_02, nv)-np.dot(ref_01, nv))
    lift_WH_gcs = np.hstack((np.array(target_width).reshape((len(target_width),1)),
                             np.array(target_height).reshape((len(target_height),1))))
    print(lift_WH_gcs.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WH_gcs.npy", lift_WH_gcs)

def get_WH_gcs_box():
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift_gcs.npy'))
    box_data = np.load(os.path.join('..', 'data', 'MOPED25', 'box_13-24_gcs_trunc.npy'))
    shelf_xyz = np.load(os.path.join('..', 'data', 'MOPED25', 'shelf.npy'))
    box_height = []
    box_width = []
    print(x_data.shape, box_data.shape)
    for i in range(len(box_data)):
        # use the distance between mid of big toes and center of box
        middle_bigT = (x_data[i, (33 - 1) * 3: (33 - 1) * 3 + 3] +
                     x_data[i, (32 - 1) * 3: (32 - 1) * 3 + 3]) / 2
        center_box = np.mean(box_data[i,:-6].reshape((-1,3)),axis=0)
        box_width.append(np.linalg.norm(middle_bigT-center_box))
        box_height.append(center_box[1])
    lift_WH_gcs = np.hstack((np.array(box_width).reshape((len(box_width), 1)),
                             np.array(box_height).reshape((len(box_height), 1))))
    print(lift_WH_gcs.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WH_gcs.npy", lift_WH_gcs)

def add_sub_height_to_WH():
    WH = np.load(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WH_gcs.npy")
    height_all = np.load(os.path.join('..', 'data', 'MOPED25', 'sub_height.npy'))
    y_all = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))

    height_temp=[]
    for i in range(len(WH)):
        height_temp.append(height_all[int(y_all[i, 0]-1)]*10)
    height_temp = np.array(height_temp).reshape((len(WH),1))
    print(height_temp.shape)
    WHH = np.hstack((WH, height_temp))
    print(WHH.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_WHH_gcs.npy", WHH)
def mean_cc_height():
    x_all = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_gcs.npy'))
    print((x_all[0, (34 - 1) * 3+1]+x_all[0, (35 - 1) * 3+1])/2)


def split_data():
    x_all = np.load(np.load(os.path.join('..','data','MOPED25','x_moped25.npy')))
    y_all = np.load(os.path.join('.','data','MOPED25','y_moped25.npy'))

def get_trunc_box():
    box_10 = np.load(os.path.join('..','data', 'MOPED25', 'box_13-24_gcs.npy'))
    box_label = np.load(os.path.join('..','data', 'MOPED25', 'box_13-24_label.npy'))
    pose_label = np.load(os.path.join('..','data', 'MOPED25', 'y_moped25_lift.npy'))
    print(box_label.shape, box_10.shape, box_10.shape)
    box_10_trunc=[]
    box_10_label_trunc=[]
    for i in range(2,13):
        for j in range(13, 25):
            for k in range(1,4):
                ttt = np.bitwise_and(box_label[:,0]==i,box_label[:,1]==j)
                box_temp = box_label[np.bitwise_and(ttt, box_label[:,2]==k),:]
                box_temp_xyz= box_10[np.bitwise_and(ttt, box_label[:,2]==k),:]
                ttt = np.bitwise_and(pose_label[:, 0] == i, pose_label[:, 1] == j)
                pose_temp = pose_label[np.bitwise_and(ttt, pose_label[:,2]==k),:]

                if len(box_temp)==0 or len(pose_temp)==0: # continue when its corrupted files
                    print(i,j,k,len(box_temp),len(pose_temp))
                    continue
                # box_temp_max = np.max(box_temp[:,3])
                # pose_temp_max = np.max(pose_temp[:,3])
                # if box_temp_max>pose_temp_max:
                #     print(box_temp_max, pose_temp_max)
                #     diff = round(box_temp_max)-round(pose_temp_max)
                if len(box_temp)>len(pose_temp):
                    diff = len(box_temp)-len(pose_temp) # get number of redundant
                else:
                    diff=0
                # change box label
                box_temp = box_temp[int(diff):, :]
                box_temp[:, -1] = np.arange(len(box_temp))+1
                # change box xyz
                box_temp_xyz = box_temp_xyz[int(diff):, :]
                box_temp_xyz[:, -1] = np.arange(len(box_temp_xyz))
                if len(box_10_trunc)==0:
                    box_10_trunc = box_temp_xyz
                    box_10_label_trunc = box_temp
                else:
                    box_10_trunc = np.vstack((box_10_trunc, box_temp_xyz))
                    box_10_label_trunc = np.vstack((box_10_label_trunc, box_temp))
    print(box_10_label_trunc.shape, box_10_trunc.shape, pose_label.shape)
    np.save(os.path.join('..','data', 'MOPED25', 'box_13-24_gcs_trunc.npy'), box_10_trunc)
    np.save(os.path.join('..', 'data', 'MOPED25', 'box_13-24_label_trunc.npy'), box_10_label_trunc)

def get_hand_on_box_label():
    box_xyz = np.load(os.path.join('..', 'data', 'MOPED25', 'box_13-24_gcs_trunc.npy'))
    box_label = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25_lift.npy'))
    pose_xyz = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift.npy'))
    diff_list=[]
    handonbox=[0]
    for i in range(1, len(box_label)):
        if round((box_label[i,3]))-round((box_label[i-1,3]))==1:
            c1=np.mean(box_xyz[i,:-6].reshape((-1,3)),axis=0)
            c2=np.mean(box_xyz[i-1,:-6].reshape((-1,3)),axis=0)
            diff_list.append(np.linalg.norm(c1-c2))
            # if np.linalg.norm(c1-c2)>
            # handonbox.append(1)
            if np.linalg.norm(c1-c2)>0.5:
                handonbox.append(1)
            else:
                handonbox.append(0)
        else:
            handonbox.append(0)
            continue
    new_box_label = np.hstack((box_label, np.array(handonbox).reshape((len(box_label),1))))
    diff_list=np.array(diff_list)
    print(new_box_label.shape)

    np.save(os.path.join('..', 'data', 'MOPED25', 'y_moped25_lift.npy'), new_box_label)
    # for i in range(0,20,1):
    #     print(np.sum(diff_list>i/10))

    # import matplotlib.pyplot as plt
    # plt.interactive(False)
    # _ = plt.hist(diff_list, bins='auto', range = (0,2))
    # plt.show()

def get_HVL_LR_gcs():
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift_gcs.npy'))
    shelf_xyz = np.load(os.path.join('..', 'data', 'MOPED25', 'shelf.npy'))

    target_v_L, target_v_R, target_h_L, target_h_R, target_l_L, target_l_R=[[] for _ in range(6)]

    for i in range(len(x_data)):
        # middle of RS and us as reference height (vertical)
        target_v_L.append((x_data[i, (15 - 1) * 3 + 1] +
                           x_data[i, (17 - 1) * 3 + 1]) / 2)
        target_v_R.append((x_data[i, (14 - 1) * 3+1] +
                        x_data[i, (16 - 1) * 3+1]) / 2)

        # use the distance difference between middle of R/L CC and wrist (Horizontal)
        middle_cc = (x_data[i, (34- 1) * 3: (34- 1) * 3 + 3] +
                       x_data[i, (35 - 1) * 3: (35- 1) * 3 + 3]) / 2
        middle_wrist_L = (x_data[i, (15 - 1) * 3: (15 - 1) * 3 + 3] +
                          x_data[i, (17 - 1) * 3: (17 - 1) * 3 + 3]) / 2
        middle_wrist_R = (x_data[i, (14- 1) * 3: (14- 1) * 3 + 3] +
                       x_data[i, (16 - 1) * 3: (16- 1) * 3 + 3]) / 2
        # vector perpendicular to the shelf
        nv = (shelf_xyz[22,:]-shelf_xyz[23,:]).astype(np.float)
        nv /= np.linalg.norm(nv) # unit normal vector
        # connect any point on the shelf to wrist
        ref_01_L = middle_wrist_L - shelf_xyz[22, :]
        ref_01_R = middle_wrist_R - shelf_xyz[22,:]
        ref_02 = middle_cc - shelf_xyz[22, :]
        # distance
        target_h_L.append(np.dot(ref_02, nv) - np.dot(ref_01_L, nv))
        target_h_R.append(np.dot(ref_02, nv) - np.dot(ref_01_R, nv))

        # use the distance between wrist and sagittal plane defined by C7, T8, IJ
        C7, IJ, T8  = [x_data[i, (jid- 1) * 3: (jid- 1) * 3 + 3] for jid in [6,7,8]]
        nv_sagittal = np.cross(C7 - T8, IJ - T8)
        nv_sagittal = nv_sagittal / np.sqrt(sum(nv_sagittal ** 2)) # unit vector

        target_l_L.append(abs(np.dot(middle_wrist_L, nv_sagittal)))
        target_l_R.append(abs(np.dot(middle_wrist_R, nv_sagittal)))

    HVL_LR_gcs = np.array([target_v_L, target_v_R, target_h_L, target_h_R, target_l_L, target_l_R]).transpose()
    print(HVL_LR_gcs.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\HVL_LR_gcs.npy", HVL_LR_gcs)

def add_sub_height_to_HVL():
    HVL = np.load(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\HVL_LR_gcs.npy")
    height_all = np.load(os.path.join('..', 'data', 'MOPED25', 'sub_height.npy'))
    y_all = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25.npy'))

    height_temp=[]
    for i in range(len(HVL)):
        height_temp.append(height_all[int(y_all[i, 0]-1)]*10)
    height_temp = np.array(height_temp).reshape((len(HVL),1))
    print(height_temp.shape)
    HHVL = np.hstack((height_temp, HVL))
    print(HHVL.shape)
    HHV = np.delete(HHVL, [5,6], axis=1)
    print(HHV.shape)
    np.save(r"F:\Research\MyPaper\IKGan\pycode\data\MOPED25\lift_HHV_gcs.npy", HHV)

def angle_dist(sub_id = [2], level_id=[], target_height=1.2, h_range=0.03):
    x_data = np.load(os.path.join('..', 'data', 'MOPED25', 'x_moped25_lift.npy'))
    z_data = np.load(os.path.join('..', 'data', 'MOPED25', 'y_moped25_lift.npy'))
    y_data = np.load(os.path.join('..', 'data', 'MOPED25', 'lift_HHV_gcs.npy'))

    # select frames with hand on box
    hob_ids = z_data[:, 4] == 1
    z_data = z_data[hob_ids, :]
    y_data = y_data[hob_ids, :]
    x_data = x_data[hob_ids, :]
    # select designated data (sub id and level id)
    selected_id = []
    if len(sub_id) > 0 or len(level_id) > 0:
        for i in range(len(z_data)):
            if len(sub_id) > 0 and len(level_id) > 0:
                if z_data[i, 0] in sub_id and z_data[i, 1] in level_id:
                    selected_id.append(i)
            elif z_data[i, 0] in sub_id or z_data[i, 1] in level_id:
                selected_id.append(i)
        selected_id = np.unique(selected_id)  # remove repeated samples
        x_data = x_data[selected_id, :]
        y_data = y_data[selected_id, :]
        z_data = z_data[selected_id, :]
    # delete identifiers
    x_data /= 1000
    y_data /= 1000  # should normalized wh as well
    x_data = np.delete(x_data, np.array([111, 110, 109, 108, 107, 106]) - 1, axis=1)  # delete identifier
    # select height
    selected_id = []
    if target_height is not None:
        for i in range(len(z_data)):
            if y_data[i,1]>target_height-h_range and y_data[i,1]<target_height+h_range:
                if y_data[i,2]>target_height-h_range and y_data[i,2]<target_height+h_range:
                    selected_id.append(i)
    x_data = x_data[selected_id, :]
    y_data = y_data[selected_id, :]
    z_data = z_data[selected_id, :]



    plt.hist(y_data[:,1],bins=100)
    plt.show()

# def model_select():
#     import glob
#     pkl_path_all = glob.glob(os.path.join('..','weights','*.pickle'))
#
#     best_loss_sum =10000
#     for pkl_path in pkl_path_all:
#         wt_dict_candidate = load_weight_pickle(pickle_path=pkl_path)
#         best_epoch = wt_dict_candidate["best epochs"]
#
#         # get mean of loss for each epoch
#         train_loss_list = np.mean(wt_dict_candidate["train loss list"], axis=1)
#         vali_loss_list = np.mean(wt_dict_candidate["validation loss list"], axis=1)
#         # select the bst one
#         best_train_loss = train_loss_list[best_epoch]
#         best_vali_loss = vali_loss_list[best_epoch]
#
#         if best_loss_sum >  best_train_loss+best_vali_loss:
#             best_loss_sum = best_train_loss + best_vali_loss
#             best_wt_dict = wt_dict_candidate
#
#     print(best_loss_sum)
#     best_weight_path = wt_dict["weight save path"]

def sensitivity_analysis():
    from data import datasets
    import metrics
    gp_metrics = metrics.gp_metrics()
    colors = ['black', 'blue', 'red','yellow','green','fuchsia',
              'grey','ivory', 'cyan','coral', 'brown']
    # load data
    heights = []
    means = []
    for sub_id in [2,3,4,5,6,7,8,9,10,11,12]:
        angles = []
        moped25 = datasets.MOPED25_lift(sub_id=[sub_id],rd="..", inter_frames=0)
        x_data, y_data, z_data = moped25.x_data, moped25.y_data, moped25.z_data
        heights+=[y_data[0,0]]
        for i in range(len(x_data)):
            gp_metrics.pose = x_data[i,:]
            angles_temp = gp_metrics.seg_angles()
            angles+=[angles_temp]
        angles_np = np.array(angles)
        # angles_mean = np.mean(np.array(angles), axis=0)
        # angles_var = np.var(np.array(angles), axis=0)
        means+=[np.mean(angles_np[:,:,1])]
    print(heights, means)
    plt.scatter(heights, means)
    plt.show()



if __name__ == "__main__":
    sensitivity_analysis()