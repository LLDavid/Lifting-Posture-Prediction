import numpy as np
import sys
import os
class gp_metrics:
    # self.joints_name = ['FH', 'R_TP', 'L_TP', 'R_ACRO', 'L_ACRO', 'C7', 'IJ', 'T8', 'PX',
    #                     'R_EL', 'L_EL', 'R_EM', 'L_EL', 'R_RS', 'L_RS', 'R_US', 'L_US',
    #                     'R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS', 'R_FE', 'L_FE', 'R_LT',
    #                     'L_LT', 'R_MT', 'L_MT', 'R_MM', 'L_MM', 'R_LM', 'L_LM', 'R_BT',
    #                     'L_BT', 'R_CC', 'L_CC']
    def __init__(self):
        self.pose=None
        np.seterr(divide='ignore', invalid='ignore')
    def id_to_vec(self, id, pose=None):
        return pose[3*(id-1):3*(id-1)+3]
    def id_distance(self, id1, id2):
        dist = np.linalg.norm(self.pose[3*(id1-1):3*(id1-1)+3]-self.pose[3*(id2-1):3*(id2-1)+3])
        return dist
    def xyz_distance(self, x1, x2):
        dist = np.linalg.norm(x1 - x2)
        return dist
    def id_mid_point(self, *ids):
        sum_temp=0
        for id in ids:
            sum_temp += self.pose[3*(id-1):3*(id-1)+3]
        return sum_temp/len(ids)
    def vec_angle(self, v1, v2):
        dot_prod = np.dot(v1 / np.linalg.norm(v1),v2 / np.linalg.norm(v2))
        if np.isnan(dot_prod):
            exit()
        return np.arccos(dot_prod)
    def planeNormal_from_3pts(self, pt1, pt2, pt3):
        n = np.cross(pt3 - pt1, pt2 - pt1)
        return n/np.sqrt(sum(n**2))
    def project_vec_to_pl(self, vec, pl_normal):
        return vec-(np.dot(vec, pl_normal) / pl_normal ** 2) * pl_normal

    def seg_lengths(self):
        # self.joints_name = ['FH', 'R_TP', 'L_TP', 'R_ACRO', 'L_ACRO', 'C7', 'IJ', 'T8', 'PX',
        #                     'R_EL', 'L_EL', 'R_EM', 'L_EL', 'R_RS', 'L_RS', 'R_US', 'L_US',
        #                     'R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS', 'R_FE', 'L_FE', 'R_LT',
        #                     'L_LT', 'R_MT', 'L_MT', 'R_MM', 'L_MM', 'R_LM', 'L_LM', 'R_BT',
        #                     'L_BT', 'R_CC', 'L_CC']
        seg_lts=[]

        # upper arm length (R/L) (# acromian to mid of EM and EL)
        ua_length_R = self.xyz_distance(self.id_to_vec(4, pose=self.pose), self.id_mid_point(10, 12))
        ua_length_L = self.xyz_distance(self.id_to_vec(5, pose=self.pose), self.id_mid_point(11, 13))
        seg_lts+=[ua_length_L, ua_length_R]

        # forearm length (R/L)
        la_length_R = self.xyz_distance(self.id_mid_point(10, 12), self.id_mid_point(14, 16))
        la_length_L=self.xyz_distance(self.id_mid_point(11, 13), self.id_mid_point(15, 17))
        seg_lts+=[la_length_L, la_length_R]

        # upper trunk length
        up_trunk_length = self.xyz_distance(self.id_mid_point(6,7), self.id_mid_point(8, 9))
        lo_trunk_length = self.xyz_distance(self.id_mid_point(8,9), self.id_mid_point(18,19,20,21))
        seg_lts += [up_trunk_length, lo_trunk_length]

        # thigh length R/L
        thi_legnth_R = self.id_distance(id1=18, id2=22)
        thi_legnth_L = self.id_distance(id1=19, id2=23)
        seg_lts += [thi_legnth_L, thi_legnth_R]

        # leg length R/L
        leg_length_R = self.xyz_distance(self.id_mid_point(28, 30), self.id_mid_point(24, 26))
        leg_length_L = self.xyz_distance(self.id_mid_point(29, 31), self.id_mid_point(25, 27))
        seg_lts += [leg_length_L,  leg_length_R]

        seg_lts = np.array(seg_lts).reshape((1, len(seg_lts)))
        return seg_lts

    def seg_angles(self):

        seg_angles=[]
        # head bending angle
        hb_ref_pl = self.planeNormal_from_3pts(self.pose[3*(6-1):3*(6-1)+3], self.pose[3*(7-1):3*(7-1)+3],
                                                   self.pose[3*(8-1):3*(8-1)+3])
        v_06_01 = self.pose[3 * (1 - 1):3 * (1 - 1) + 3] - self.pose[3 * (6 - 1):3 * (6 - 1) + 3]
        v_06_07 = self.pose[3 * (7 - 1):3 * (7 - 1) + 3] - self.pose[3 * (6 - 1):3 * (6 - 1) + 3]
        # v_06_01_p = self.project_vec_to_pl(vec=v_06_01, pl_normal=hb_ref_pl)
        # v_06_07_p = self.project_vec_to_pl(vec=v_06_07, pl_normal=hb_ref_pl)
        hb_angle = self.vec_angle(v1=v_06_01, v2=v_06_07)
        seg_angles.append(hb_angle)

        # trunk bending
        v_08_06 = self.pose[3 * (6 - 1):3 * (6 - 1) + 3] - self.pose[3 * (8 - 1):3 * (8 - 1) + 3]
        v_tb_ref = self.id_mid_point(18, 19) - self.id_mid_point(20, 21)
        tb_angle = self.vec_angle(v_08_06, v_tb_ref)
        seg_angles.append(tb_angle)

        # upperarm ext/flex (L/R)
        uaef_ref_pl = self.planeNormal_from_3pts(self.pose[3*(6-1):3*(6-1)+3], self.pose[3*(7-1):3*(7-1)+3],
                                                   self.pose[3*(8-1):3*(8-1)+3])
        v_ua_L = self.id_mid_point(11, 13) - self.pose[3 * (5 - 1):3 * (5 - 1) + 3]
        v_ua_p_L = self.project_vec_to_pl(vec=v_ua_L, pl_normal=uaef_ref_pl)
        v_06_08 = self.pose[3 * (8 - 1):3 * (8 - 1) + 3] - self.pose[3 * (6 - 1):3 * (6 - 1) + 3]
        v_06_08_p = self.project_vec_to_pl(vec=v_06_08, pl_normal=uaef_ref_pl)
        uaef_angle_L = self.vec_angle(v_ua_p_L, v_06_08_p)
        seg_angles.append(uaef_angle_L)


        v_ua_R = self.id_mid_point(10, 12)-self.pose[3 * (4 - 1):3 * (4 - 1) + 3]
        v_ua_p_R = self.project_vec_to_pl(vec=v_ua_R, pl_normal=uaef_ref_pl)
        v_06_08 = self.pose[3 * (8 - 1):3 * (8 - 1) + 3] - self.pose[3 * (6 - 1):3 * (6 - 1) + 3]
        v_06_08_p = self.project_vec_to_pl(vec=v_06_08, pl_normal=uaef_ref_pl)
        uaef_angle_R = self.vec_angle(v_ua_p_R, v_06_08_p)
        seg_angles.append(uaef_angle_R)

        # forearm ext/flex (L/R)
        v_la = self.id_mid_point(15, 17) - self.id_mid_point(11, 13)
        ualf_angle_L = self.vec_angle(v_la, v_ua_L)
        seg_angles.append(ualf_angle_L)
        v_la = self.id_mid_point(14, 16)-self.id_mid_point(10, 12)
        ualf_angle_R = self.vec_angle(v_la, v_ua_R)
        seg_angles.append(ualf_angle_R)

        # leg ext/flex (L/R)
        v_19_23 = self.pose[3 * (23 - 1):3 * (23 - 1) + 3] - self.pose[3 * (19 - 1):3 * (19 - 1) + 3]
        v_leg_ref_L = self.pose[3 * (23 - 1):3 * (23 - 1) + 3] - self.id_mid_point(29, 31)
        lgef_angle_L = self.vec_angle(v_19_23, v_leg_ref_L)
        seg_angles.append(lgef_angle_L)

        v_18_22 = self.pose[3 * (22 - 1):3 * (22 - 1) + 3] - self.pose[3 * (18 - 1):3 * (18 - 1) + 3]
        v_leg_ref_R =  self.pose[3 * (22 - 1):3 * (22 - 1) + 3] - self.id_mid_point(28, 30)
        lgef_angle_R = self.vec_angle(v_18_22, v_leg_ref_R)
        seg_angles.append(lgef_angle_R)

        # dorsiflexion and plantarflexion
        v_foot_ref_L = self.pose[3 * (33 - 1):3 * (33 - 1) + 3] - self.id_mid_point(29, 31)
        dp_angle_L = self.vec_angle(v_leg_ref_L, v_foot_ref_L)
        seg_angles.append(dp_angle_L)
        v_foot_ref_R = self.pose[3 * (32 - 1):3 * (32 - 1) + 3] - self.id_mid_point(28, 30)
        dp_angle_R = self.vec_angle(v_leg_ref_R, v_foot_ref_R)
        seg_angles.append(dp_angle_R)
        seg_angles = np.array(seg_angles).reshape((1, len(seg_angles)))/np.pi*180
        return seg_angles

    def vec_to_metrics(self, v1=None, v2=None, mtype="rmse", HHW=None):
        if mtype == "rmse":
            return np.sqrt(np.square(v1.reshape((105,))-v2.reshape((105,))).mean())
        if mtype == "mae":
            return np.abs(v1.reshape((105,)) - v2.reshape((105,))).mean()
        if mtype == "3drmse":
            v1 = np.reshape(v1, (int(len(v1) / 3), 3))
            v2 = np.reshape(v2, (int(len(v2) / 3), 3))
            joint_temp = []
            for i in range(len(v1)):
                joint_temp += [np.square(np.linalg.norm(v1[i,:])-np.linalg.norm(v2[i,:]))]
            return np.sqrt(np.mean(joint_temp))
        if mtype == "3dmae":
            v1 = np.reshape(v1, (int(len(v1) / 3), 3))
            v2 = np.reshape(v2, (int(len(v2) / 3), 3))
            joint_temp = []
            for i in range(len(v1)):
                joint_temp  += [np.abs(np.linalg.norm(v1[i,:])-np.linalg.norm(v2[i,:]))]
            return np.mean(joint_temp)
        if mtype == "HW":
            shelf_xyz = np.load(os.path.join('.', 'data', 'MOPED25', 'shelf.npy'))
            # H
            middle_cc = (v1[(34 - 1) * 3: (34 - 1) * 3 + 3] +
                         v1[(35 - 1) * 3: (35 - 1) * 3 + 3]) / 2
            # middle of RS and us as reference height (vertical)
            H_L=(v1[(15 - 1) * 3 + 1] +v1[(17 - 1) * 3 + 1]) / 2-middle_cc[1]+0.045
            H_R=(v1[(14 - 1) * 3 + 1] +v1[(16 - 1) * 3 + 1]) / 2-middle_cc[1]+0.045
            # W
            middle_wrist_L = (v1[(15 - 1) * 3: (15 - 1) * 3 + 3] +
                              v1[(17 - 1) * 3: (17 - 1) * 3 + 3]) / 2
            middle_wrist_R = (v1[(14 - 1) * 3: (14 - 1) * 3 + 3] +
                              v1[(16 - 1) * 3: (16 - 1) * 3 + 3]) / 2
            middle_cc = (v1[(34 - 1) * 3: (34 - 1) * 3 + 3] +
                         v1[(35 - 1) * 3: (35 - 1) * 3 + 3]) / 2
            # vector perpendicular to the shelf
            nv = (shelf_xyz[22, :] - shelf_xyz[23, :]).astype(np.float)
            nv /= np.linalg.norm(nv)  # unit normal vector
            # connect any point on the shelf to wrist
            ref_01_L = middle_wrist_L - shelf_xyz[22, :]
            ref_01_R = middle_wrist_R - shelf_xyz[22, :]
            ref_02 = middle_cc - shelf_xyz[22, :]
            # distance
            W_L=np.dot(ref_02, nv) - np.dot(ref_01_L, nv)
            W_R=np.dot(ref_02, nv) - np.dot(ref_01_R, nv)

            return [(np.abs(HHW[1]-H_L)+np.abs(HHW[2]-H_R))/2,
                   (np.abs(HHW[3]-W_L)+np.abs(HHW[4]-W_R))/2]

        if mtype == "angles":
            self.pose=v1
            v1_angles = self.seg_angles()
            self.pose = v2
            v2_angles = self.seg_angles()

            abs_angles = np.abs(np.array(v1_angles)-np.array(v2_angles))
            rmse_angles =  np.sqrt(np.square(np.array(v1_angles)-np.array(v2_angles)).mean())

            return abs_angles, rmse_angles