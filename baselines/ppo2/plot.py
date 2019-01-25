import time, os, platform
import traceback

import pandas as pd
import numpy as np

# if platform.system() == "Windows":
#     import matplotlib.pyplot as plt
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# else:
#     pass
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import seaborn as sns


class AttentionPlot(object):
    def __init__(self, limited, save_path):
        self.plot_dir = os.path.join("/home/netease/data/save/baseline/plot/%s" % save_path)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.heatmap_count = 0
        self.limited = limited
        self.text_file_name = os.path.join(self.plot_dir, "state_action.csv")
        self.attention_file_name = os.path.join(self.plot_dir, "attention.csv")
        self.history_attention_file_name = os.path.join(self.plot_dir, "history_attention.csv")

    def write_text_log(self, str):
        with open(self.text_file_name, "a") as writer:
            writer.write(str + "\n")

    def write_history_attention_log(self, str):
        with open(self.history_attention_file_name, "a") as writer:
            writer.write(str + "\n")

    def remove_current_attention_file(self):
        try:
            os.remove(self.attention_file_name)
        except Exception as e:
            traceback.print_exc()

    def write_attention_log(self, str):
        with open(self.attention_file_name, "a") as writer:
            writer.write(str + "\n")

    def heatmap(self, df, save=False, show=False, file_name=""):
        # print("[AttentionPlot] plot a heatmap.")
        if isinstance(np.array(df), pd.DataFrame):
            pass
        else:
            df = pd.DataFrame(df)
        f, ax = plt.subplots(figsize=(10, 6), dpi=200)
        # cmap = sns.cubehelix_palette(n_colors=1, start=1, rot=3, gamma=0.8, as_cmap=True)
        cmap = sns.color_palette("Blues",n_colors=100)
        sns.heatmap(df, xticklabels=1, yticklabels=1, cmap=cmap, linewidths=0.05, ax=ax)
        ax.set_title('Attention for different actions on states')
        ax.set_xlabel('state')
        ax.set_ylabel('action')
        plt.xticks(fontsize=8, color="black", rotation=90)
        plt.yticks(fontsize=8, color="black", rotation=0)
        if show and ((self.heatmap_count < self.limited and self.limited > 0) or self.limited <= 0):
            plt.show()
        if save:
            if ((self.heatmap_count < self.limited and self.limited > 0) or self.limited <= 0):
                # print("[%s] Start to save the attention image." % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
                try:
                    f.savefig(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count) + '.jpg', bbox_inches='tight')
                except ValueError as e:
                    f.savefig(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count) + '.png', bbox_inches='tight')
                # print("[%s] Complete to save the attention image." % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            elif self.heatmap_count >= self.limited:
                try:
                    os.remove(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count - self.limited) + '.jpg')
                    f.savefig(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count) + '.jpg', bbox_inches='tight')
                except Exception as e:
                    os.remove(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count - self.limited) + '.png')
                    f.savefig(self.plot_dir + "/" + 'attention_heatmap_' + str(self.heatmap_count) + '.png', bbox_inches='tight')
        self.heatmap_count += 1
        plt.close()
        # print("[%s] Leave the plot function." % time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    def offline_heatmap(self, filepath, state_dim=16):
        df = pd.read_csv(filepath_or_buffer=filepath, header=None)
        nbatch = df.values.shape[0]
        print(df.values.shape)
        attention_array = np.reshape(df.values, newshape=(-1, state_dim))
        attention_array_list = np.split(attention_array, indices_or_sections=nbatch, axis=0)
        df_sum = attention_array_list[0]
        for idx, element in enumerate(attention_array_list):
            if idx >= 1:
                df_sum = df_sum + element
        df_mean = df_sum / nbatch
        index = ["thigh_joint", "leg_joint", "foot_joint"]
        columns = []
        # df = pd.DataFrame(df_mean, index=index)
        df = pd.DataFrame(df_mean)
        f, ax = plt.subplots(figsize=(10, 6), dpi=200)
        # cmap = sns.cubehelix_palette(n_colors=1, start=1, rot=3, gamma=0.8, as_cmap=True)
        cmap = sns.color_palette("Blues",n_colors=100)
        sns.heatmap(df, xticklabels=1, yticklabels=1, cmap=cmap, linewidths=0.05,annot=True, ax=ax)
        ax.set_title('Attention for different actions on states')
        ax.set_xlabel('state')
        ax.set_ylabel('action')
        plt.xticks(fontsize=8, color="black", rotation=90)
        plt.yticks(fontsize=8, color="black", rotation=0)
        f.savefig("/".join(filepath.split("/")[:-1]) + "/" + 'attention_heatmap_mean.png', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    filepath = r"E:/Experiments/ActionAttention/201901/Walker2d-v2_Attention_2019_01_25_00_08_06_Sigmoid-False_entcoef-0.0_lr0.0003_loss-fix_clip_weak/attention.csv"
    plot = AttentionPlot(10, "")
    plot.offline_heatmap(filepath=filepath, state_dim=17)
