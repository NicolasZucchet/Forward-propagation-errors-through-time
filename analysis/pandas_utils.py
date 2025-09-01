import pandas as pd


def select(df, selection):
    new_df = df.copy()
    for key, value in selection.items():
        if isinstance(value, list):
            new_df = new_df[new_df[key].isin(value)]
        else:
            new_df = new_df[new_df[key] == value]
    return new_df


def select_last_step(df, groupby, seed="seed", step="_step", val_loss="val/loss"):
    # For each groupby x seed, pick the last step for which the loss is not nan
    # Remove all lines with nan training loss
    df = df[df[val_loss].notna()]
    groups = df.groupby(groupby + [seed])
    last_steps = []
    for name, group in groups:
        last_step = group.loc[group[step].idxmax()]
        last_steps.append(last_step)
    return pd.DataFrame(last_steps)


def select_best_lr(df, keys, best_lrs, lr="training.learning_rate"):
    dfs = []
    for this_keys, this_lr in best_lrs.items():
        selection = {lr: this_lr}
        for key, this_key in zip(keys, this_keys):
            selection[key] = this_key
        dfs.append(select(df, selection))
    return pd.concat(dfs)


def find_best_lr(df, groupby, lr="training.learning_rate", metric="val/loss", step="_step"):
    # For each combinations in groupby, average the results, and find the best learning rate, the one that minimize the metric at the final time step
    df = select_last_step(df, groupby + [lr])
    groups = df.groupby(groupby)
    best_lrs = {}
    for name, group in groups:
        best_lr = group.loc[group[metric].idxmin(), lr]
        best_lrs[name] = best_lr
    return best_lrs
