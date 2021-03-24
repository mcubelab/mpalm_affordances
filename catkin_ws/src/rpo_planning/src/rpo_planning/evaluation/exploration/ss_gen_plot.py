import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(font_scale=2.0)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

def plot1():
    data = np.load("grasping_ss_pos_data.npz")
    gat_joint_mask = data['gat_joint_mask']
    pointnet_joint_mask = data['pointnet_joint_mask']

    gat_indep_mask = data['gat_indep_mask']
    pointnet_indep_mask = data['pointnet_indep_mask']

    gat_joint_trans = data['gat_joint_trans']
    gat_indep_trans = data['gat_indep_trans']

    gat_joint_mask_err = data['gat_joint_mask_err']
    pointnet_joint_mask_err = data['pointnet_joint_mask_err']

    gat_indep_mask_err = data['gat_indep_mask_err']
    pointnet_indep_mask_err = data['pointnet_indep_mask_err']

    gat_joint_trans_err = data['gat_joint_trans_err']
    gat_indep_trans_err = data['gat_indep_trans_err']    

    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT", "GAT"], 
    # "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans", "Indep \nTrans"], 
    # "Value": [gat_joint_mask[0], pointnet_joint_mask[0], gat_indep_mask[0], pointnet_indep_mask[0], gat_joint_trans[0], gat_indep_trans[0]]})

    df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT"], 
    "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans"], 
    "Value": [gat_joint_mask[0], pointnet_joint_mask[0], gat_indep_mask[0], pointnet_indep_mask[0], gat_joint_trans[0]]})    

    # from IPython import embed
    # embed()

    # # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT"], 
    # # "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans"], 
    # # "Value": [[np.random.normal(gat_joint_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)], 
    # #           [np.random.normal(pointnet_joint_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)],
    # #           [np.random.normal(gat_indep_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)],
    # #           [np.random.normal(pointnet_indep_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)],
    # #           [np.random.normal(gat_joint_trans[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)]]})
    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT"], 
    # "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans"], 
    # "Value": [np.random.normal(gat_joint_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000), 
    #           np.random.normal(pointnet_joint_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000),
    #           np.random.normal(gat_indep_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000),
    #           np.random.normal(pointnet_indep_mask[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000),
    #           np.random.normal(gat_joint_trans[0], np.abs(gat_joint_mask_err[0] - gat_joint_mask_err[1])*2, 1000)]
    #           })    

    # embed()

    # g = sns.catplot(x="Name", y="Value", hue="", data=df,
    #                 height=6, kind="bar", palette="muted", legend=False)
    g = sns.catplot(x="Name", y="Value", hue="", data=df,
                    height=6, kind="bar", palette="muted", legend=False)    

    g.set_ylabels("Position Error (m)")
    g.set_xlabels("Model")
    g.set(ylim=(0, 0.05))
    # g.set(ylim=(0, 40))
    plt.title("Grasping Position Error")
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig("grasp_pos.pdf")

    # Pulling Data
    # data = np.load("pulling_ss_data.npz")

    # gat_joint_trans = data['gat_joint_trans']
    # pointnet_joint_trans = data['pointnet_joint_trans']

    # gat_indep_trans = data['gat_indep_trans']
    # pointnet_indep_trans = data['pointnet_indep_trans']

    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++"], "Name": ["Joint Trans", "Joint Trans", "Indep Trans", "Indep Trans"], "Value": [gat_joint_trans[0], pointnet_joint_trans[0], gat_indep_trans[0], pointnet_indep_trans[0]]})

    # g = sns.catplot(x="Name", y="Value", hue="", data=df,
    #                 height=6, kind="bar", palette="muted", legend=False)

    # g.set_ylabels("Percent Success")
    # g.set_xlabels("Model")
    # g.set(ylim=(0, 100))
    # plt.title("Pulling Success Rate")
    # plt.tight_layout()

    # plt.savefig("pull.pdf")


    # # Pushing Data
    # data = np.load("pushing_ss_data.npz")

    # gat_joint_trans = data['gat_joint_trans']
    # pointnet_joint_trans = data['pointnet_joint_trans']

    # gat_indep_trans = data['gat_indep_trans']
    # pointnet_indep_trans = data['pointnet_indep_trans']

    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++"], "Name": ["Joint Trans", "Joint Trans", "Indep Trans", "Indep Trans"], "Value": [gat_joint_trans[0], pointnet_joint_trans[0], gat_indep_trans[0], pointnet_indep_trans[0]]})

    # g = sns.catplot(x="Name", y="Value", hue="", data=df,
    #                 height=6, kind="bar", palette="muted", legend=False)

    # g.set_ylabels("Percent Success")
    # g.set_xlabels("Model")
    # g.set(ylim=(0, 100))
    # plt.title("Pushing Success Rate")
    # plt.tight_layout()

    # plt.savefig("push.pdf")


def plot2():
    # data = np.load("grasping_ss_grasping_data.npz")
    data = np.load("grasping_ss_mp_success_data.npz")
    gat_joint_mask = data['gat_joint_mask']
    pointnet_joint_mask = data['pointnet_joint_mask']

    gat_indep_mask = data['gat_indep_mask']
    pointnet_indep_mask = data['pointnet_indep_mask']

    gat_joint_trans = data['gat_joint_trans']
    gat_indep_trans = data['gat_indep_trans']

    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT", "GAT"], 
    # "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans", "Indep \nTrans"], 
    # "Value": [gat_joint_mask, pointnet_joint_mask, gat_indep_mask, pointnet_indep_mask, gat_joint_trans, gat_indep_trans]})

    df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++", "GAT"], 
    "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask", "Joint \nTrans"], 
    "Value": [gat_joint_mask, pointnet_joint_mask, gat_indep_mask, pointnet_indep_mask, gat_joint_trans]})

    # df = pd.DataFrame({"": ["GAT", "PointNet++", "GAT", "PointNet++"], 
    # "Name": ["Joint \nMask", "Joint \nMask", "Indep \nMask", "Indep \nMask"], 
    # "Value": [gat_joint_mask, pointnet_joint_mask, gat_indep_mask, pointnet_indep_mask]})    

    g = sns.catplot(x="Name", y="Value", hue="", data=df, n_boot=1000,
                    height=6, kind="bar", palette="muted", legend=False)

    g.set_ylabels("Percent Success")
    g.set_xlabels("Model")
    g.set(ylim=(0, 100))
    # plt.title("Sticking Success Rate")
    plt.title("Feasibility Success Rate")    
    # plt.legend(loc='lower left')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()


    # plt.savefig("grasp_grasp_success.pdf")
    plt.savefig("grasp_mp_success.pdf")


if __name__ == "__main__":
    plot1()
    # plot2()