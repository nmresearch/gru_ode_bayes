import argparse
import gru_ode_bayes.data_utils_bond as data_utils
import gru_ode_bayes
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Running GRUODE on Double OU")
parser.add_argument('--model_name', type=str, help="Model to use", default="bond_gru_ode_bayes")
parser.add_argument('--dataset', type=str, help="Dataset CSV file", default="../../gru_ode_bayes/datasets/bond/bond.csv")
parser.add_argument('--jitter', type=float, help="Time jitter to add (to split joint observations)", default=0)
parser.add_argument('--seed', type=int, help="Seed for data split generation", default=432)
parser.add_argument('--full_gru_ode', action="store_true", default=True)
parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")
parser.add_argument('--no_impute',action="store_true", default = True)
parser.add_argument('--demo', action = "store_true", default = False)

args = parser.parse_args()
#if args.demo:
#    print(f"Demo Mode - Loading model for double_OU ....")
#    gru_ode_bayes.paper_plotting.plot_trained_model(model_name = "double_OU_gru_ode_bayes_demo")
#    exit()

model_name = args.model_name


device = "cpu"
#gpu_num = 2
#device  = torch.device(f"cuda:{gpu_num}")
#torch.cuda.set_device(gpu_num)

#Dataset metadata
#metadata = np.load(f"{args.dataset[:-4]}_metadata.npy",allow_pickle=True).item()

#metadata = {"delta_t": 0.05, "T": 10, "N": 10000, "theta": 1.0, "sigma": 0.1, "r_mu": [1.0, -1.0], "sample_rate": 2, "dual_sample_rate": 0.2, "rho": 0.99, "max_lag": 0, "r_std": 1/np.sqrt(12)}
#delta_t = metadata["delta_t"]
#T       = metadata["T"]
#
#train_idx, val_idx = train_test_split(np.arange(metadata["N"]),test_size=0.2, random_state=args.seed)


import datetime
import pytz

def _format_date(x):
    return datetime.datetime.strptime(x, "%Y-%m-%d").replace(
        tzinfo=pytz.UTC
    )



#T_begin = _format_date("2019-10-01")
#T_val = (_format_date("2020-01-01")-T_begin).total_seconds()


#train_dataset = "../../gru_ode_bayes/datasets/bond/train_filter.csv"
all_dataset = "../../gru_ode_bayes/datasets/bond/all_filter.csv"
#test_dataset = "../../gru_ode_bayes/datasets/bond/test_filter.csv"
import pandas as pd
root_dir="./"

#train = pd.read_csv(root_dir + "/" + train_dataset, parse_dates=["trade_timestamp"], date_parser=lambda col: pd.to_datetime(col, utc=True),)
#train = train.rename(columns = {"trade_timestamp": "Time", "price": "Value_1", "yield": "Value_2", "Mask": "Mask_1"})
#T_begin = train["Time"].min()
#T_val = (train["Time"].max()-T_begin).total_seconds()//3600
#train["Time"] = (train["Time"]-T_begin).dt.total_seconds()//3600


all = pd.read_csv(root_dir + "/" + all_dataset, parse_dates=["trade_timestamp"], date_parser=lambda col: pd.to_datetime(col, utc=True),)
all = all.rename(columns = {"trade_timestamp": "Time", "spread": "Value_1", "yield": "Value_2", "Mask": "Mask_1", "size": "Size"})
T_begin = all.Time.min()
#all = all.loc[(all["Time"]>=T_begin) & (all["Time"]<=_format_date("2019-12-31"))]
T_val = (_format_date("2019-08-01")-T_begin).total_seconds()//1800
all["Time_accurate"] = (all["Time"]-T_begin).dt.total_seconds()/1800
all["Time"] = (all["Time"]-T_begin).dt.total_seconds()//1800

all = all.sort_values(by="Time_accurate")
all = all.groupby(["cusip", "Time"]).tail(1)


train = all.loc[all["Time"]<=T_val]
test = all.loc[all["Time"]>T_val]
#test = pd.read_csv(root_dir + "/" + test_dataset, parse_dates=["trade_timestamp"], date_parser=lambda col: pd.to_datetime(col, utc=True),)
#test = test.rename(columns = {"trade_timestamp": "Time", "price": "Value_1", "yield": "Value_2", "Mask": "Mask_1"})
#test["Time"] = (test["Time"]-T_begin).dt.total_seconds()//1800

val_options = {"T_val": T_val, "max_val_samples": 10000}

id2Int = pd.Series(np.arange(len(all.cusip.unique())), index = all.cusip.unique())
all["ID"] = id2Int[all.cusip.values].values
train["ID"] = id2Int[train.cusip.values].values
test["ID"] = id2Int[test.cusip.values].values
all=all.drop(columns=["cusip"])[["ID", "Time", "Value_1", "Mask_1", "Time_accurate", "Size"]]
train=train.drop(columns=["cusip"])[["ID", "Time", "Value_1", "Mask_1", "Time_accurate", "Size"]]
test=test.drop(columns=["cusip"])[["ID", "Time", "Value_1", "Mask_1", "Time_accurate", "Size"]]

#train_idx, val_idx = train_test_split(np.arange(all.ID.nunique()),test_size=0.2, random_state=args.seed)


# impute missing testing bonds in train
df_beforeIdx = train.ID.unique()
df_afterIdx  = test.ID.unique()
#new_indices = np.setdiff1d(df_afterIdx, df_beforeIdx)
#for i in new_indices:
#    all = all.append({'ID': i, "Value_1": 100.0, "Mask_1": 1.0, "Time": (_format_date("2020-01-01")-T_begin).total_seconds()//1800-1, "Time_accurate": (_format_date("2020-01-01")-T_begin).total_seconds()/1800-1}, ignore_index=True)

all = all[all.ID.isin(np.intersect1d(df_afterIdx, df_beforeIdx))]
id2Int = pd.Series(np.arange(len(all.ID.unique())), index = all.ID.unique())
all["ID"] = id2Int[all.ID.values].values

id2Int = pd.Series(np.arange(len(train.ID.unique())), index = train.ID.unique())
train["ID"] = id2Int[train.ID.values].values

# convert to percentages
all["Value_1"] = all["Value_1"]/100
train["Value_1"] = train["Value_1"]/100
train_mean = train["Value_1"].mean()
train_std = train["Value_1"].std()
print ("mean and std: ", train_mean, train_std)
all["Value_1"] = (all["Value_1"]-train_mean)/train_std
train["Value_1"] = (train["Value_1"]-train_mean)/train_std

all["LogSize"] = np.log(all["Size"])
train["LogSize"] = np.log(train["Size"])

data_train = data_utils.ODE_Dataset(panda_df=train)
data_val   = data_utils.ODE_Dataset(panda_df=all, validation = True,
                                    val_options = val_options )
#data_train = data_utils.ODE_Dataset(csv_file=args.dataset, idx=train_idx, jitter_time=args.jitter)
#data_val   = data_utils.ODE_Dataset(csv_file=args.dataset, idx=val_idx, jitter_time=args.jitter,validation = True,
#                                    val_options = val_options )



#delta_t = metadata["delta_t"]
#T       = metadata["T"]

T = all.Time.max()
delta_t  = 1.0
metadata = {"delta_t": 1.0, "T": T, "N": 10000, "theta": 1.0, "sigma": 0.1, "r_mu": [1.0, -1.0], "sample_rate": 2, "dual_sample_rate": 0.2, "rho": 0.99, "max_lag": 0, "r_std": 1/np.sqrt(12)}

#Model parameters.
params_dict=dict()
params_dict["input_size"]  = 1
params_dict["hidden_size"] = 50
params_dict["p_hidden"]    = 25
params_dict["prep_hidden"] = 10
params_dict["logvar"]      = True
params_dict["mixing"]      = 0.0001
params_dict["delta_t"]     = delta_t
#params_dict["dataset"]     = args.dataset
#params_dict["jitter"]      = args.jitter
#params_dict["gru_bayes"]   = "masked_mlp"
params_dict["full_gru_ode"] = True
params_dict["solver"]      = "euler"
params_dict["impute"]      = False

params_dict["T"]           = T

#Model parameters and the metadata of the dataset used to train the model are stored as a single dictionnary.
#summary_dict ={"model_params":params_dict,"metadata":metadata}
#np.save(f"./../trained_models/{model_name}_params.npy",summary_dict)

dl     = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=512, num_workers=3)
dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size=len(data_val),num_workers=2)

## the neural negative feedback with observation jumps
model = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                        p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                        logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                        full_gru_ode = params_dict["full_gru_ode"],
                                        solver = params_dict["solver"], impute = params_dict["impute"])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
epoch_max = 50


#Training
for epoch in range(epoch_max):
    model.train()
    for i, b in tqdm.tqdm(enumerate(dl)):
        optimizer.zero_grad()
        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(device)
        M        = b["M"].to(device)
        size     = b["size"].to(device)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(device)
        y = b["y"]
        print ("train")
        hT, loss, _, _  = model(times, time_ptr, X, M, obs_idx, size, delta_t=delta_t, T=T, cov=cov)
        print ("loss", loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mse_val  = 0
        mae_val = 0
        weighted_mae_val = 0
        loss_val = 0
        num_obs  = 0
        model.eval()
        for i, b in enumerate(dl_val):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            size     = b["size"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            X_val     = b["X_val"].to(device)
            M_val     = b["M_val"].to(device)
            size_val  = b["size_val"].to(device)
            times_val = b["times_val"]
            times_idx = b["index_val"]
            y = b["y"]
            print ("eval")
            hT, loss, _, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx, size, delta_t=delta_t, T=T, cov=cov, return_path=True)
            t_vec = np.around(t_vec,str(delta_t)[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
            p_val     = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
            m, v      = torch.chunk(p_val,2,dim=1)
            print ("X_val", X_val, "obs_idx", obs_idx, "m", m, "v", v)
            print (m[:30])
            last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
            print ("last_loss", last_loss)
            mse_loss  = (torch.pow(X_val - m, 2) * M_val).sum() * train_std**2
            print ("mse_loss", mse_loss)
            mae_loss =  (torch.abs(X_val - m) * M_val).sum() * train_std
            print ("mae_loss", mae_loss)
            num_obs  += M_val.sum().cpu().numpy()
            weighted_mae_loss = (torch.abs(X_val - m) * M_val * size_val).sum() * train_std * num_obs/size_val.sum()
            print ("weighted_mae_loss", weighted_mae_loss)
            loss_val += last_loss.cpu().numpy()
            mse_val  += mse_loss.cpu().numpy()
            mae_val  += mae_loss.cpu().numpy()
            weighted_mae_val += weighted_mae_loss.cpu().numpy()
            print (b["df_val"])
        loss_val /= num_obs
        mse_val  /= num_obs
        mae_val /= num_obs
        weighted_mae_val /= num_obs
        print(f"Mean validation loss at epoch {epoch}: nll={loss_val:.5f}, mse={mse_val:.5f}, mae={mae_val:.5f}, weighted_mae={weighted_mae_val:.5f}, (num_obs={num_obs})")

import pickle
with open(f"results_yield_pct", "wb") as f:
    pickle.dump({"X_val": X_val, "obs_idx": obs_idx, "m": m, "v": v, "df_val": df_val}, f)


print(f"Last validation log likelihood : {loss_val}")
print(f"Last validation MSE : {mse_val}")
print(f"Last validation MAE : {mae_val}")
df_file_name = "./../trained_models/bond_results.csv"
df_res = pd.DataFrame({"Name" : [model_name], "LogLik" : [loss_val], "MSE" : [mse_val], "MAE": [mae_val], "Dataset": [args.dataset], "Seed": [args.seed]})
if os.path.isfile(df_file_name):
    df = pd.read_csv(df_file_name)
    df = df.append(df_res)
    df.to_csv(df_file_name,index=False)
else:
    df_res.to_csv(df_file_name,index=False)


model_file = f"./../trained_models/{model_name}.pt"
torch.save(model.state_dict(),model_file)
print(f"Saved model into '{model_file}'.")


"""
Plotting resulting model on newly generated_data
"""
gru_ode_bayes.paper_plotting.plot_trained_model(model_name = model_name)
