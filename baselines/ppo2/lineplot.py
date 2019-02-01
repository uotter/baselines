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


def lineplot_errorband(data, xname, yname, hue, title_name, show=False, save=False, game="mujoco"):
    # Plot the responses for different events and regions
    sns.set()
    f, ax = plt.subplots(figsize=(10, 6), dpi=200)
    g = sns.lineplot(x=xname, y=yname, data=data, hue=hue)
    # fig = data.plot().get_figure()
    plt.title(title_name, fontsize=23)
    plt.legend(loc='lower right', fontsize=23)
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
            mode = filename.split("_")[2] if filename.split("_")[2] == "Attention" else "PPO"
            if mode == "Attention":
                activation = "jump" if len(filename.split("_")) >= 15 else "concat"
                # activation = filename.split("_")[9].split("-")[0] if filename.split("_")[9].split("-")[1] == "True" else "Softmax"
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
            if sigmoid:
                dataname = "Action" + mode + "_" + activation
            else:
                dataname = "Action" + mode
        else:
            # dataname = env + "_" + mode
            dataname = mode
        # if not sigmoid and activation == "Sigmoid":
        if not sigmoid and activation == "concat":
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


def merge_result_csv_for_sns(dir_path, sigmoid=False, max_step=2e8, game="atari"):
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
        if dataname == "ActionAttention_jump" and not sigmoid:
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


def get_average_from_result(dir_path, sigmoid=False, max_step=2e8, game="atari"):
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
        if dataname == "ActionAttention_concat" and not sigmoid:
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
                # raw_df = pd.read_csv(complete_sub_file_path, header=None, sep='\t', engine="python", names=columns)
                # for index, row in raw_df.iterrows():
                #     if row["length"] is None:
                #         print(row)
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
                raw_df = raw_df.sort_values(by="step",ascending=False)
                raw_df = raw_df.iloc[:1000,:]
                print("{} {} {}: {}".format(env,dataname,raw_df["step"].values[0],raw_df["reward"].mean()))
    return result_df_dic

if __name__ == "__main__":
    game = ["atari", "mujoco"][0]
    source = ["tb", "result"][1]
    max_step = 2e8
    sigmoid = False
    show = False
    value_or_plot = "value"
    if value_or_plot == "value":
        if source == "tb":
            raise NotImplementedError("value option with tb data not implemented.")
        else:
            dir_path = r"E:\Experiments\ActionAttention\result\{}".format(game)
            get_average_from_result(dir_path, sigmoid=sigmoid, max_step=max_step, game=game)
    else:
        if source == "tb":
            dir_path = r"E:\Experiments\ActionAttention\{}".format(game)
            result_df_dic = merge_mujuco_tb_csv_for_sns(dir_path, sigmoid=sigmoid, max_step=max_step, game=game)
        else:
            dir_path = r"E:\Experiments\ActionAttention\result\{}".format(game)
            result_df_dic = merge_result_csv_for_sns(dir_path, sigmoid=sigmoid, max_step=max_step, game=game)
        for k, v in result_df_dic.items():
            print(k)
            print(v)
            if source == "tb":
                if game == "mujoco":
                    lineplot_errorband(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True,
                                       show=show,
                                       game=game)
                elif game == "atari":
                    lineplot_tb(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=show,
                                game=game, weight=0.9)
            else:
                lineplot_errorband(xname="step", yname="reward", hue="method", data=v, title_name=k, save=True, show=show,
                                   game=game)
            # break
