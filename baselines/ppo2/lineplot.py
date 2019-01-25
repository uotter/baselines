import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

sns.set(style="darkgrid")


def smooth(scalars, weight):
    '''
    tesnorbaord like smooth method
    :param scalars: real scalar values in a list
    :param weight:smooth weight between 0 and 1, the bigger means heave smooth
    :return:
    '''
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def lineplot_tb(data, xname, yname, hue, title_name, show=False, save=False, game="mujoco", weight=0.6):
    # Plot the responses for different events and regions
    sns.set()
    f, ax = plt.subplots(figsize=(10, 6), dpi=200)
    # g = sns.lineplot(x=xname, y=yname, data=data, hue=hue)
    method_name_set = set(list(data[hue].values))
    method_name_list, xs_list, ys_list, smooth_ys_list = [], [], [], []
    for method_name in method_name_set:
        sub_data = data[data[hue] == method_name]
        sub_data = sub_data.sort_values(by="step")
        xs_list.append(list(sub_data["step"].values))
        ys_list.append(list(sub_data["reward"].values))
        method_name_list.append(method_name)
        smooth_ys_list.append(smooth(list(sub_data["reward"].values), weight=weight))
    colors = ["coral","steelblue"]
    for idx, method_name in enumerate(method_name_list):
        plt.plot(xs_list[idx], ys_list[idx],alpha=0.1,color=colors[idx])
        plt.plot(xs_list[idx], smooth_ys_list[idx], label=method_name,color=colors[idx])
    # fig = data.plot().get_figure()
    plt.title(title_name)
    plt.legend(loc='lower right')
    if not os.path.exists("./plot/{}/".format(game)):
        os.makedirs("./plot/{}".format(game))
    if save:
        # fig.savefig("./plot/" + title_name + ".png")
        f.savefig("./plot/{}/".format(game) + title_name + ".png")
    if show:
        plt.show()


def lineplot_errorband(data, xname, yname, hue, title_name, show=False, save=False, game="mujoco"):
    # Plot the responses for different events and regions
    sns.set()
    f, ax = plt.subplots(figsize=(10, 6), dpi=200)
    g = sns.lineplot(x=xname, y=yname, data=data, hue=hue)
    # fig = data.plot().get_figure()
    plt.title(title_name)
    plt.legend(loc='lower right')
    if not os.path.exists("./plot/{}/".format(game)):
        os.makedirs("./plot/{}".format(game))
    if save:
        # fig.savefig("./plot/" + title_name + ".png")
        f.savefig("./plot/{}/".format(game) + title_name + ".png")
    if show:
        plt.show()


def merge_mujuco_tb_csv(dir_path, sigmoid=False, max_step=1e7, game="mujoco"):
    files = os.listdir(dir_path)  # 得到文件夹下的所有文件名称
    result_df_dic = {}
    for file in files:  # 遍历文件夹
        complete_file_path = os.path.join(dir_path, file)
        if os.path.isdir(complete_file_path):
            continue
        print(complete_file_path)
        (filepath, tempfilename) = os.path.split(file)
        (filename, extension) = os.path.splitext(tempfilename)
        env = filename.split("_")[1]
        mode = filename.split("_")[2] if filename.split("_")[2] == "Attention" else "ppo"
        if game == "mujoco":
            env = filename.split("_")[1]
            mode = filename.split("_")[2] if filename.split("_")[2] == "Attention" else "ppo"
            if mode == "Attention":
                activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
            else:
                activation = ""
        elif game == "atari":
            env = filename.split("_")[2]
            mode = "Attention" if filename.split("_")[6] != "normal" else "impala"
            if mode == "Attention":
                activation = filename.split("_")[14] if filename.split("_")[15] == "True" else "Softmax"
            else:
                activation = ""
        else:
            raise NotImplementedError("Not implemented for game {}".format(game))
        if mode == "Attention":
            # dataname = env + "_" + mode + "_" + activation
            if sigmoid:
                dataname = "Action" + mode + "_" + activation
            else:
                dataname = "Action" + mode
        else:
            # dataname = env + "_" + mode
            dataname = mode
        if not sigmoid and activation == "Sigmoid":
            continue
        if not os.path.isdir(complete_file_path):
            current_df = pd.read_csv(complete_file_path)
            current_df = current_df.iloc[:, 1:]
            current_df.columns = ["step", dataname]
            current_df = current_df[current_df["step"] < max_step]
            current_df = current_df.set_index("step")
            # print(current_df)
            if env in result_df_dic.keys():
                result_df = result_df_dic[env]
                result_df = pd.concat([result_df, current_df], axis=1)
            else:
                result_df = current_df
            result_df_dic[env] = result_df
    for k, v in result_df_dic.items():
        if sigmoid:
            result_df_dic[k] = result_df_dic[k][["ActionAttention_Softmax", "ActionAttention_Sigmoid", "ppo"]]
        else:
            result_df_dic[k] = result_df_dic[k][["ActionAttention", "ppo"]]
    return result_df_dic


def merge_mujuco_tb_csv_for_sns(dir_path, sigmoid=False, max_step=1e7, game="mujoco"):
    files = os.listdir(dir_path)  # 得到文件夹下的所有文件名称
    result_df_dic = {}
    for file in files:  # 遍历文件夹
        complete_file_path = os.path.join(dir_path, file)
        if os.path.isdir(complete_file_path):
            continue
        print(complete_file_path)
        (filepath, tempfilename) = os.path.split(file)
        (filename, extension) = os.path.splitext(tempfilename)
        if game == "mujoco":
            env = filename.split("_")[1]
            mode = filename.split("_")[2] if filename.split("_")[2] == "Attention" else "ppo"
            if mode == "Attention":
                activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
            else:
                activation = ""
        elif game == "atari":
            env = filename.split("_")[2]
            mode = "Attention" if filename.split("_")[6] != "normal" else "impala"
            if mode == "Attention":
                activation = filename.split("_")[14] if filename.split("_")[15] == "True" else "Softmax"
            else:
                activation = ""
        else:
            raise NotImplementedError("Not implemented for game {}".format(game))

        if mode == "Attention":
            # dataname = env + "_" + mode + "_" + activation
            if sigmoid:
                dataname = "Action" + mode + "_" + activation
            else:
                dataname = "Action" + mode
        else:
            # dataname = env + "_" + mode
            dataname = mode
        if not sigmoid and activation == "Sigmoid":
            continue
        if not os.path.isdir(complete_file_path):
            current_df = pd.read_csv(complete_file_path)
            dataname_list = [dataname] * len(current_df)
            dataname_se = pd.Series(dataname_list)
            current_df.insert(loc=3, column="method", value=dataname_se)
            current_df = current_df.iloc[:, 1:]
            current_df.columns = ["step", "reward", "method"]
            current_df = current_df[current_df["step"] < max_step]
            # current_df = current_df.set_index("step")
            # print(current_df)
            if env in result_df_dic.keys():
                result_df = result_df_dic[env]
                result_df = pd.concat([result_df, current_df], axis=0)
            else:
                result_df = current_df
            result_df_dic[env] = result_df
    return result_df_dic


if __name__ == "__main__":
    game = ["atari", "mujoco"][1]
    dir_path = r"E:\Experiments\ActionAttention\{}".format(game)
    max_step = 5e7
    result_df_dic = merge_mujuco_tb_csv_for_sns(dir_path, sigmoid=False, max_step=max_step, game=game)
    for k, v in result_df_dic.items():
        print(k)
        print(v)
        if game == "mujoco":
            lineplot_errorband(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=True, game=game)
        elif game=="atari":
            lineplot_tb(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=True, game=game, weight=0.6)
