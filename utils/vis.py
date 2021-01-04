import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d

class pose_visualizer:
    def __init__(self):
        self.link_pair = np.array([[1, 2], [1, 3], [2, 3], [1, 6], [4, 6], [5, 6],
                                [4, 7], [5, 7], [6, 7], [6, 8], [7, 8], [6, 9],
                                [7, 9], [8, 9], [4, 10], [5, 11], [12, 4], [10, 12],
                                [5, 13], [13, 11], [10, 14], [12, 14], [11, 15], [13, 15],
                                [10, 16], [16, 12], [14, 16], [17, 11], [17, 13], [17, 15],
                                [18, 9], [19, 9], [19, 18], [20, 8], [20, 18], [20, 19],
                                [21, 8], [21, 18], [19, 21], [21, 20], [18, 22], [19, 23],
                                [18, 24], [22, 24], [19, 25], [23, 25], [26, 18], [22, 26],
                                [24, 26], [27, 19], [27, 23], [25, 27], [22, 28], [26, 28],
                                [23, 29], [27, 29], [22, 30], [24, 30], [28, 30], [23, 31],
                                [25, 31], [29, 31], [28, 32], [32, 30], [29, 33], [31, 33],
                                [34, 28], [34, 30], [32, 34], [29, 35], [31, 35], [33, 35]])-1
        self.joints_name = ['FH', 'R_TP', 'L_TP', 'R_ACRO', 'L_ACRO', 'C7','IJ', 'T8', 'PX',
                            'R_EL', 'L_EL', 'R_EM', 'L_EL', 'R_RS', 'L_RS', 'R_US', 'L_US',
                            'R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS', 'R_FE', 'L_FE', 'R_LT',
                            'L_LT', 'R_MT', 'L_MT', 'R_MM', 'L_MM', 'R_LM', 'L_LM', 'R_BT',
                            'L_BT', 'R_CC', 'L_CC']

        self.subpose_id = np.array([1, 4, 5, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 32, 33, 34, 35])-1
        self.subpose_link_pair=np.array([[1,4], [2,4], [3,4], [2,5], [3,5], [2,6],
                                [3,7], [7,9], [2,6], [6,8], [2,10], [3, 11],
                                [10,12], [10,11], [11,13], [12,14], [12,16], [13, 15],
                                [13, 17], [14, 16], [15, 17]])-1
    # for visualize 3d pose (all joints)
    def pose37_3d(self, pose_all):
        # create axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pid in range(len(pose_all)):
            print("The "+str(pid)+" th pose. ")
            if pose_all.ndim>1:
                pose_i=pose_all[pid,:]
                n_joints = int(pose_all.shape[1]/3)
            else:
                pose_i=pose_all
                n_joints = int(len(pose_all)/3)
            # get all x y z separately
            pose_x = pose_i[[ii * 3 for ii in range(n_joints)]]
            pose_y = pose_i[[ii * 3 + 1 for ii in range(n_joints)]]
            pose_z = pose_i[[ii * 3 + 2 for ii in range(n_joints)]]
            # plot points
            ax.scatter(pose_x, pose_y, pose_z, c='r', marker='o')
            # add joints name
            for i, txt in enumerate(self.joints_name):
                # ax.annotate(txt, pose_x[i], pose_y[i], pose_z[i])
                ax.text(pose_x[i], pose_y[i], pose_z[i], txt)
            # plot lines
            for i in range(len(self.link_pair)):
                line_xs = [pose_x[self.link_pair[i, ii]] for ii in range(2)]
                line_ys = [pose_y[self.link_pair[i, ii]] for ii in range(2)]
                line_zs = [pose_z[self.link_pair[i, ii]] for ii in range(2)]

                #l_color = np.random.rand(3)
                #line = plt3d.art3d.Line3D(line_xs, line_ys, line_zs,
                                         # color=l_color)# random color
                line = plt3d.art3d.Line3D(line_xs, line_ys, line_zs)# default color
                ax.add_line(line)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.axis('off')
            # for va1 in range(0,360,10):
            #     for va2 in (0,360,10):
            #         fig.canvas.set_window_title(str(va1)+' '+str(va2))
            #         ax.view_init(va1,va2)
            #         plt.draw()
            #         # plt.pause(0.01)
            #         plt.savefig('fig_'+str(va1)+' '+str(va2)+'.png',
            #                     orientation='portrait')
            # exit(0)
            ax.view_init(340, 360)
            plt.draw()
            if pose_all.ndim>1:
                plt.pause(0.1)
                plt.cla()
            else:
                plt.pause(10)
                break
        return plt
    def pose37_3d_4gan(self, pose_all, ax):

            pose_all = pose_all
            # get all x y z separately
            pose_x = pose_all[[ii * 3 for ii in range(int(len(pose_all)/3))]]
            pose_y = pose_all[[ii * 3 + 1 for ii in range(int(len(pose_all)/3))]]
            pose_z = pose_all[[ii * 3 + 2 for ii in range(int(len(pose_all)/3))]]
            # plot points
            ax.scatter(pose_x, pose_y, pose_z, c='r', marker='o')
            for i, txt in enumerate(self.joints_name):
                ax.text(pose_x[i], pose_y[i], pose_z[i], txt)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # plot lines
            for i in range(len(self.link_pair)):
                line_xs = [pose_x[self.link_pair[i, ii]] for ii in range(2)]
                line_ys = [pose_y[self.link_pair[i, ii]] for ii in range(2)]
                line_zs = [pose_z[self.link_pair[i, ii]] for ii in range(2)]
                line = plt3d.art3d.Line3D(line_xs, line_ys, line_zs)
                ax.add_line(line)

    def image_2D(self, img_rows=800, img_cols=600, pose_all=0, sub = True):

        # scale a little to make it look nicer
        img_rows-=5
        img_cols-=5

        pose_xy=pose_all.reshape(37,3)
        x_range = np.max(pose_xy[:, 0])-np.min(pose_xy[:, 0]) # get the range of xys
        y_range = np.max(pose_xy[:, 1]) - np.min(pose_xy[:, 1])

        if sub is True:
            x=img_rows-(pose_xy[self.subpose_id, 0]-np.min(pose_xy[:, 0]))/x_range*img_rows
            y=img_cols-(pose_xy[self.subpose_id, 1]-np.min(pose_xy[:, 1]))/y_range*img_cols
            link_pair_temp = self.subpose_link_pair # lines
            joints_name_temp=list(map(self.joints_name.__getitem__, self.subpose_id))
            print(joints_name_temp)
        else:
            x = img_rows-(pose_xy[:, 0] - np.min(pose_xy[:, 0])) / x_range * img_rows
            y = img_cols-(pose_xy[:, 1] - np.min(pose_xy[:, 1])) / y_range * img_cols
            link_pair_temp = self.link_pair
            joints_name_temp=self.joints_name

        img = np.ones((img_rows, img_cols,3))
        ax = plt.gca()
        plt.imshow(img)

        # plot points
        plt.scatter(x=x, y=y, c='r', s=40)

        # add text
        for i, txt in enumerate(joints_name_temp):
            # ax.annotate(txt, pose_x[i], pose_y[i], pose_z[i])
            print(i, txt)
            ax.text(x[i], y[i], txt)

        # plot lines
        for i in range(len(self.subpose_link_pair)):
            line_xs = [x[link_pair_temp[i, ii]] for ii in range(2)]
            line_ys = [y[link_pair_temp[i, ii]] for ii in range(2)]
            line = plt.Line2D(line_xs,line_ys)
            ax.add_line(line)

        plt.axis('on')
        plt.show()

    def plot_two_37(self, p1, p2, text=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        n_joints = int(len(p1)/3)
        # get all x y z separately

        pose_x1 = p1[[ii * 3 for ii in range(n_joints)]]
        pose_y1 = p1[[ii * 3 + 1 for ii in range(n_joints)]]
        pose_z1 = p1[[ii * 3 + 2 for ii in range(n_joints)]]

        pose_x2 = p2[[ii * 3 for ii in range(n_joints)]]
        pose_y2 = p2[[ii * 3 + 1 for ii in range(n_joints)]]
        pose_z2 = p2[[ii * 3 + 2 for ii in range(n_joints)]]
        # plot points
        ax.scatter(pose_x1, pose_y1, pose_z1, c='g', marker='o')
        ax.scatter(pose_x2, pose_y2, pose_z2, c='y', marker='o')
        ax.view_init(elev=87, azim=270)
        # # add joints name
        # for i, txt in enumerate(self.joints_name):
        #     # ax.annotate(txt, pose_x[i], pose_y[i], pose_z[i])
        #     ax.text(pose_x[i], pose_y[i], pose_z[i], txt)
        # plot lines
        for i in range(len(self.link_pair)):
            line_xs1 = [pose_x1[self.link_pair[i, ii]] for ii in range(2)]
            line_ys1 = [pose_y1[self.link_pair[i, ii]] for ii in range(2)]
            line_zs1 = [pose_z1[self.link_pair[i, ii]] for ii in range(2)]

            line_xs2 = [pose_x2[self.link_pair[i, ii]] for ii in range(2)]
            line_ys2 = [pose_y2[self.link_pair[i, ii]] for ii in range(2)]
            line_zs2 = [pose_z2[self.link_pair[i, ii]] for ii in range(2)]

            # l_color = np.random.rand(3)
            # line = plt3d.art3d.Line3D(line_xs, line_ys, line_zs,
            # color=l_color)# random color
            line1 = plt3d.art3d.Line3D(line_xs1, line_ys1, line_zs1)  # default color
            line2 = plt3d.art3d.Line3D(line_xs2, line_ys2, line_zs2, ls='--', color='r')
            ax.add_line(line1)
            ax.add_line(line2)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        plt.axis('off')
            # for va1 in range(0,360,10):
            #     for va2 in (0,360,10):
            #         fig.canvas.set_window_title(str(va1)+' '+str(va2))
            #         ax.view_init(va1,va2)
            #         plt.draw()
            #         # plt.pause(0.01)
            #         plt.savefig('fig_'+str(va1)+' '+str(va2)+'.png',
            #                     orientation='portrait')
            # exit(0)
        # ax.view_init(340, 360)
        if text is not None:
            plt.title(text)
        plt.draw()


        plt.pause(0.01)
        # plt.clf()

        return plt






# if __name__ == "__main__":
#    # x = np.load(r'F:\Research\MyPaper\IKGan\pycode\data\MOPED25\x_moped25_mini.npy')
#    # a=pose_visualizer()
#    # a.pose37_3d(pose_all=x)