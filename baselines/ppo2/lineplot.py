import datetime
import platform

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

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
    for method_name in sorted(list(method_name_set)):
        sub_data = data[data[hue] == method_name]
        sub_data = sub_data.sort_values(by="step")
        xs_list.append(list(sub_data["step"].values))
        ys_list.append(list(sub_data["reward"].values))
        method_name_list.append(method_name)
        smooth_ys_list.append(smooth(list(sub_data["reward"].values), weight=weight))
    colors = ["steelblue", "coral", "green"]
    for idx, method_name in enumerate(method_name_list):
        plt.plot(xs_list[idx], ys_list[idx], alpha=0.1, color=colors[idx])
        plt.plot(xs_list[idx], smooth_ys_list[idx], label=method_name, color=colors[idx])
    # fig = data.plot().get_figure()
    plt.title(title_name, fontsize=23)
    plt.legend(loc='lower right', fontsize=23)
    plt.xlabel("step", fontsize=23)
    plt.ylabel("reward", fontsize=23)
    if not os.path.exists("./plot/{}/".format(game)):
        os.makedirs("./plot/{}".format(game))
    if save:
        # fig.savefig("./plot/" + title_name + ".png")
        f.savefig("./plot/{}/".format(game) + title_name + ".png")
    if show:
        plt.show()


def lineplot_errorband(data, xname, yname, hue, title_name, show=False, save=False, game="mujoco", paper_used=False):
    # Plot the responses for different events and regions
    sns.set()
    f, ax = plt.subplots(figsize=(10, 6), dpi=200)
    # Shrink current axis by 20%
    box = ax.get_position()
    if not paper_used:
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    hue_list = sorted(list(set(data[hue].values)))
    g = sns.lineplot(x=xname, y=yname, data=data, hue=hue, hue_order=hue_list)
    # fig = data.plot().get_figure()
    plt.title(title_name, fontsize=23)
    if paper_used:
        plt.legend(loc='lower right', fontsize=23)
    else:
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5))
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
                activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[
                                                                         1] == "True" else "Softmax"
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


def merge_mujuco_tb_csv_for_sns(dir_path, selected_labels=[], max_step=1e7, game="mujoco"):
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
            mode = "PPO" if filename.split("_")[2] == "NoAttention" else filename.split("_")[2]
            if "loss-fix" in filename:
                if mode == "Attention":
                    if len(filename.split("_")) >= 15:
                        if "-" in filename.split("_")[14]:
                            activation = (filename.split("_")[14]).split("-")[0]
                        else:
                            activation = filename.split("_")[14]
                    else:
                        activation = "concat"
                    # activation = filename.split("_")[14] if len(filename.split("_")) >= 15 else "concat"
                    # activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
                elif mode == "StateAttention":
                    activation = (filename.split("_")[10]).split("-")[1]
                else:
                    activation = ""
            elif "loss-name-fix" in filename:
                if mode == "Attention":
                    activation = filename.split("_")[9].split("-")[1]
                    jump = filename.split("_")[17].split("-")[1]
                    concat = filename.split("_")[18].split("-")[1]
                    activation = activation + "_" + jump + "_" + concat
                    # activation = filename.split("_")[14] if len(filename.split("_")) >= 15 else "concat"
                    # activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
                elif mode == "StateAttention":
                    activation = filename.split("_")[9].split("-")[1]
                    jump = filename.split("_")[17].split("-")[1]
                    concat = filename.split("_")[18].split("-")[1]
                    activation = activation + "_" + jump + "_" + concat
                else:
                    activation = ""
            else:
                if mode == "StateAttention" and len(filename.split("_")) > 10:
                    activation = (filename.split("_")[10]).split("-")[1]
                else:
                    activation = ""
        elif game == "atari":
            env = filename.split("_")[2]
            mode = "Attention" if filename.split("_")[6] != "normal" else "IMPALA"
            if mode == "Attention":
                activation = filename.split("_")[16] if filename.split("_")[17] == "True" else "concat"
            else:
                activation = ""
        else:
            raise NotImplementedError("Not implemented for game {}".format(game))

        if mode == "Attention":
            # dataname = env + "_" + mode + "_" + activation
            if len(selected_labels) == 1:
                dataname = "Action" + mode
            else:
                dataname = "Action" + mode + "_" + activation
        elif activation == "":
            dataname = mode
        else:
            # dataname = env + "_" + mode
            dataname = mode + "_" + activation
        # if not sigmoid and activation == "Sigmoid":
        if len(selected_labels) > 0 and activation not in selected_labels:
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


def merge_result_csv_for_sns(dir_path, selected_labels=[], max_step=2e8, game="atari"):
    files = os.listdir(dir_path)  # 得到文件夹下的所有文件名称
    result_df_dic = {}
    for file in files:  # 遍历文件夹
        complete_file_path = os.path.join(dir_path, file)
        if not os.path.isdir(complete_file_path):
            continue
        print(complete_file_path)
        if game == "atari":
            env = file.split("_")[1]
        else:
            env = file.split("_")[0]
        sub_files = os.listdir(complete_file_path)
        dataname = "IMPALA"
        for sub_file in sub_files:
            complete_sub_file_path = os.path.join(complete_file_path, sub_file)
            (filepath, tempfilename) = os.path.split(sub_file)
            (filename, extension) = os.path.splitext(tempfilename)
            if filename == "config":
                for line in open(complete_sub_file_path):
                    if line.startswith('PACKAGE = ["baseline", "jump", "concat"]'):
                        dataname_index = int(line[-3])
                        if dataname_index == 1:
                            dataname = "ActionAttention_jump"
                        elif dataname_index == 2:
                            dataname = "ActionAttention_concat"
                break
        if len(selected_labels) > 0 and len(dataname.split("_")) >= 2 and not dataname.split("_")[1] in selected_labels:
            if len(selected_labels) == 1:
                dataname = "ActionAttention"
            continue
        for sub_file in sub_files:
            complete_sub_file_path = os.path.join(complete_file_path, sub_file)
            (filepath, tempfilename) = os.path.split(sub_file)
            (filename, extension) = os.path.splitext(tempfilename)
            f_float = lambda x: float(x.split(":")[-1])
            f_int = lambda x: int(x.split(":")[-1])
            if sub_file.startswith("test"):
                line_columns = open(complete_sub_file_path).readline().split("\t")
                columns = ["model_index", line_columns[1].split(":")[0].replace("total_", ""),
                           line_columns[2].split(":")[0].replace("total_", ""), "raw_step"]
                raw_df = pd.read_csv(complete_sub_file_path, header=None, sep='\t', engine="python", names=columns,
                                     converters={"model_index": lambda x: int(x.split(":")[0]), "reward": f_float,
                                                 "length": f_float, "raw_step": f_int})
                raw_df = raw_df.sort_values(by="model_index")
                # print(raw_df.loc[raw_df.groupby(by="model_index")["step"].idxmin(),"step"])
                raw_df['step'] = raw_df.groupby('model_index')['raw_step'].transform(lambda x: x.min())
                dataname_list = [dataname] * len(raw_df)
                dataname_se = pd.Series(dataname_list)
                raw_df.insert(loc=3, column="method", value=dataname_se)
                raw_df = raw_df[raw_df["step"] < max_step]
                if env in result_df_dic.keys():
                    result_df = result_df_dic[env]
                    result_df = pd.concat([result_df, raw_df], axis=0)
                else:
                    result_df = raw_df
                result_df_dic[env] = result_df
    return result_df_dic


def merge_logger_csv_for_sns(start, end, dir_path, selected_labels=[], envs=[], max_step=2e8, game="atari", paper_used=False):
    files = os.listdir(dir_path)  # 得到文件夹下的所有文件名称
    result_df_dic = {}
    datanames_list = []
    for file in files:  # 遍历文件夹
        complete_file_path = os.path.join(dir_path, file)
        mtime = time.localtime(os.path.getmtime(complete_file_path))
        ctime = time.ctime(os.path.getctime(complete_file_path))
        # if not os.path.isdir(complete_file_path) or mtime < time.strptime("2019-01-24-00-00-00", "%Y-%m-%d-%H-%M-%S"):
        if not os.path.isdir(complete_file_path):
            continue
        # print(complete_file_path)
        (filepath, tempfilename) = os.path.split(file)
        # (filename, extension) = os.path.splitext(tempfilename)
        filename = tempfilename
        if game == "mujoco":
            env = file.split("_")[0]
            if env not in envs:
                continue
            mode = "PPO" if filename.split("_")[1] == "NoAttention" else filename.split("_")[1]
            print(filename)
            # if mode != filename.split("_")[2]:
            #     logger_time = time.strptime("-".join(filename.split("_")[3:9]), "%Y-%m-%d-%H-%M-%S")
            # else:
            #     logger_time = time.strptime("-".join(filename.split("_")[2:8]), "%Y-%m-%d-%H-%M-%S")
            if "loss-fix" in filename:
                if paper_used:
                    continue
                logger_time = time.strptime("-".join(filename.split("_")[2:8]), "%Y-%m-%d-%H-%M-%S")
                if mode == "Attention":
                    activation = "softmax" if filename.split("_")[8].split("-")[1] == "False" else "sigmoid"
                    if len(filename.split("_")) >= 14:
                        if "-" in filename.split("_")[13]:
                            jump = (filename.split("_")[13]).split("-")[0]
                        else:
                            jump = filename.split("_")[13]
                    else:
                        jump = "concat"
                    if jump == "concat":
                        concat = "concat"
                        jump = "nojump"
                    elif jump == "residual":
                        concat = "add"
                        jump = "nojump"
                    elif jump == "residual.jump":
                        concat = "add"
                        jump = "jump"
                    else:
                        concat = "concat"
                        jump = "jump"
                    activation = activation + "_" + jump + "_" + concat
                    # activation = filename.split("_")[14] if len(filename.split("_")) >= 15 else "concat"
                    # activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
                elif mode == "StateAttention":
                    activation = (filename.split("_")[10]).split("-")[1]
                else:
                    activation = ""
            elif "loss-name-fix" in filename:
                logger_time = time.strptime("-".join(filename.split("_")[2:8]), "%Y-%m-%d-%H-%M-%S")
                if mode == "Attention":
                    activation = filename.split("_")[8].split("-")[1]
                    jump = filename.split("_")[16].split("-")[1]
                    concat = filename.split("_")[17].split("-")[1]
                    activation = activation + "_" + jump + "_" + concat
                    # activation = filename.split("_")[14] if len(filename.split("_")) >= 15 else "concat"
                    # activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
                elif mode == "StateAttention":
                    activation = filename.split("_")[8].split("-")[1]
                    jump = filename.split("_")[16].split("-")[1]
                    concat = filename.split("_")[17].split("-")[1]
                    activation = activation + "_" + jump + "_" + concat
                else:
                    activation = ""
            else:
                logger_time = time.strptime("-".join(filename.split("_")[3:9]), "%Y-%m-%d-%H-%M-%S")
                if mode == "StateAttention" and len(filename.split("_")) >= 10:
                    activation = (filename.split("_")[9]).split("-")[1]
                else:
                    activation = ""
        elif game == "atari":
            env = file.split("_")[1]
            mode = "Attention" if filename.split("_")[6] != "normal" else "IMPALA"
            if mode == "Attention":
                activation = filename.split("_")[16] if filename.split("_")[17] == "True" else "concat"
            else:
                activation = ""
        else:
            raise NotImplementedError("Not implemented for game {}".format(game))
        if logger_time is None or logger_time > end or logger_time < start:
            continue
        if mode == "Attention":
            # dataname = env + "_" + mode + "_" + activation
            if len(selected_labels) == 1:
                dataname = "Action" + mode
            else:
                dataname = "Action" + mode + "_" + activation
        elif activation == "":
            dataname = mode
        else:
            # dataname = env + "_" + mode
            dataname = mode + "_" + activation
        if len(selected_labels) > 0 and dataname not in selected_labels:
            continue
        if dataname not in datanames_list:
            datanames_list.append(dataname)
        if paper_used:
            if "StateAttention" in dataname:
                dataname = "StateAttention"
            elif "ActionAttention" in dataname:
                dataname = "ActionAttention"
            else:
                pass
        sub_files = os.listdir(complete_file_path)
        for sub_file in sub_files:
            complete_sub_file_path = os.path.join(complete_file_path, sub_file)
            (filepath, tempfilename) = os.path.split(sub_file)
            (filename, extension) = os.path.splitext(tempfilename)
            if filename == "progress":
                data = pd.read_csv(complete_sub_file_path)
                dataname_list = [dataname] * len(data)
                dataname_se = pd.Series(dataname_list)
                data.insert(loc=0, column="method", value=dataname_se)
                data = data[["serial_timesteps", "eprewmean", "eplenmean", "method"]]
                data.columns = ["step", "reward", "length", "method"]
                raw_df = data[data["step"] < max_step]
                if env in result_df_dic.keys():
                    result_df = result_df_dic[env]
                    result_df = pd.concat([result_df, raw_df], axis=0)
                else:
                    result_df = raw_df
                result_df_dic[env] = result_df
                break
    for i in datanames_list:
        print(i)
    return result_df_dic


def get_average_from_result(dir_path, selected_labels=[], max_step=2e8, game="atari"):
    files = os.listdir(dir_path)  # 得到文件夹下的所有文件名称
    result_df_dic = {}
    for file in files:  # 遍历文件夹
        complete_file_path = os.path.join(dir_path, file)
        if not os.path.isdir(complete_file_path):
            continue
        print(complete_file_path)
        if game == "atari":
            env = file.split("_")[1]
        else:
            env = file.split("_")[0]
        sub_files = os.listdir(complete_file_path)
        # if env != "SpaceInvaders":
        #     continue
        dataname = "IMPALA"
        for sub_file in sub_files:
            complete_sub_file_path = os.path.join(complete_file_path, sub_file)
            (filepath, tempfilename) = os.path.split(sub_file)
            (filename, extension) = os.path.splitext(tempfilename)
            if filename == "config":
                for line in open(complete_sub_file_path):
                    if line.startswith('PACKAGE = ["baseline", "jump", "concat"]'):
                        dataname_index = int(line[-3])
                        if dataname_index == 1:
                            dataname = "ActionAttention_jump"
                        elif dataname_index == 2:
                            dataname = "ActionAttention_concat"
                break
        if len(selected_labels) > 0 and len(dataname.split("_")) >= 2 and not dataname.split("_")[1] in selected_labels:
            if len(selected_labels) == 1:
                dataname = "ActionAttention"
            continue
        for sub_file in sub_files:
            complete_sub_file_path = os.path.join(complete_file_path, sub_file)
            (filepath, tempfilename) = os.path.split(sub_file)
            (filename, extension) = os.path.splitext(tempfilename)
            f_float = lambda x: float(x.split(":")[-1])
            f_int = lambda x: int(x.split(":")[-1])
            if sub_file.startswith("test"):
                line_columns = open(complete_sub_file_path).readline().split("\t")
                columns = ["model_index", line_columns[1].split(":")[0].replace("total_", ""),
                           line_columns[2].split(":")[0].replace("total_", ""), "raw_step"]
                raw_df = pd.read_csv(complete_sub_file_path, header=None, sep='\t', engine="python", names=columns)
                for index, row in raw_df.iterrows():
                    if row["length"] is None:
                        print(index,row)
                raw_df = pd.read_csv(complete_sub_file_path, header=None, sep='\t', engine="python", names=columns,
                                     converters={"model_index": lambda x: int(x.split(":")[0]), "reward": f_float,
                                                 "length": f_float, "raw_step": f_int})
                raw_df = raw_df.sort_values(by="model_index")
                # print(raw_df.loc[raw_df.groupby(by="model_index")["step"].idxmin(),"step"])
                raw_df['step'] = raw_df.groupby('model_index')['raw_step'].transform(lambda x: x.min())
                dataname_list = [dataname] * len(raw_df)
                dataname_se = pd.Series(dataname_list)
                raw_df.insert(loc=3, column="method", value=dataname_se)
                raw_df = raw_df[raw_df["step"] < max_step]
                raw_df = raw_df.sort_values(by="step", ascending=False)
                raw_df = raw_df.iloc[:1000, :]
                print("{} {} {}: {}".format(env, dataname, raw_df["step"].values[0], raw_df["reward"].mean()))
    return result_df_dic


if __name__ == "__main__":
    game = ["atari", "mujoco"][0]
    source = ["tb", "result", "logger"][1]
    max_step = 2e8
    paper_used = True
    if platform.system().lower() == "windows":
        show = False
    else:
        show = False
    value_or_plot = ["value", "plot"][0]
    envs = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Reacher-v2", "Swimmer-v2", "Walker2d-v2", "Humanoid-v2"]
    # envs = ["Ant-v2"]
    # selected_labels = ["ActionAttention_sigmoid_jump_add", "ActionAttention_sigmoid_jump_concat",
    #                    "StateAttention_sigmoid_jump_add", "StateAttention_sigmoid_jump_concat",
    #                    "StateAttention_softmax_jump_add", "StateAttention_softmax_jump_concat",
    #                    "ActionAttention_softmax_jump_concat", "ActionAttention_softmax_nojump_concat",
    #                    "ActionAttention_softmax_jump_add",
    #                    "PPO"]
    # selected_labels = [
    #     "StateAttention_softmax_jump_concat",
    #     "ActionAttention_softmax_jump_concat",
    #     "PPO"]
    # selected_labels = ["ActionAttention_sigmoid_jump_add",
    #                    "StateAttention_sigmoid_jump_add",
    #                    "StateAttention_softmax_jump_add",
    #                    "ActionAttention_softmax_jump_add",
    #                    "PPO"]
    # selected_labels = ["ActionAttention_softmax_jump_concat",
    #                    "StateAttention_softmax_jump_concat",
    #                    "StateAttention_softmax_jump_add",
    #                    "ActionAttention_softmax_jump_add",
    #                    "PPO"]
    selected_labels = []
    start = time.strptime("2019-01-25-00-00-00", "%Y-%m-%d-%H-%M-%S")
    end = time.strptime("2019-02-16-00-00-00", "%Y-%m-%d-%H-%M-%S")
    if value_or_plot == "value":
        if source == "tb":
            raise NotImplementedError("value option with tb data not implemented.")
        else:
            dir_path = r"E:\Experiments\ActionAttention\result\{}".format(game)
            get_average_from_result(dir_path, selected_labels=selected_labels, max_step=max_step, game=game)
    else:
        if source == "tb":
            dir_path = r"E:\Experiments\ActionAttention\{}".format(game)
            result_df_dic = merge_mujuco_tb_csv_for_sns(dir_path, selected_labels=selected_labels, max_step=max_step, game=game)
        elif source == "logger":
            if platform.system().lower() == "windows":
                dir_path = r"E:\Experiments\ActionAttention\{}\logger".format(game)
            else:
                dir_path = r"/home/netease/data/save/baseline/logger"
            result_df_dic = merge_logger_csv_for_sns(start, end, dir_path, selected_labels=selected_labels, envs=envs, max_step=max_step, game=game, paper_used=paper_used)
        else:
            dir_path = r"E:\Experiments\ActionAttention\result\{}".format(game)
            result_df_dic = merge_result_csv_for_sns(dir_path, selected_labels=selected_labels, max_step=max_step, game=game)

        for k, v in result_df_dic.items():
            print(k)
            print(v)
            if source == "tb" or source == "logger":
                if game == "mujoco":
                    lineplot_errorband(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True,
                                       show=show,
                                       game=game + "_" + source, paper_used=paper_used)
                elif game == "atari":
                    lineplot_tb(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=show,
                                game=game + "_" + source, weight=0.9)
            else:
                lineplot_errorband(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=show,
                                   game=game + "_" + source, paper_used=paper_used)
            # break
