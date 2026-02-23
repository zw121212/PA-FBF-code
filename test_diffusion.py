from types import SimpleNamespace

import torch

from compute import *
from torch.utils.data import TensorDataset
import pandas as pd
from flow_models import *
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
device = 'cpu'


def generate_observations(u, n_sensors, r):
    num_traj, T, Ns = u.shape
    sensor_idx = torch.linspace(0, Ns - 1, n_sensors).long()
    u_obs = u[:, :, sensor_idx]
    noise = torch.randn_like(u_obs) * torch.sqrt(torch.tensor(r, dtype=u.dtype))
    Y = torch.exp(-u_obs + 0.5) + noise
    return Y, sensor_idx


sigma = 1
N = 10; T = 100
r2 = 1
# r = torch.sqrt(torch.tensor(r2))  # 0.25,0.5,0.75,1.0 观测噪声：在不同强度观测噪声的情况下测试方法
dim_u = 50; dim_y = N; num_layers=6; hidden_dim=64
df1 = pd.read_csv("Data/diffusion_equation/u_train_Ns={}_T={}_sigma={}.csv".format(dim_u, T,sigma))
U_train = torch.tensor(df1.values, dtype=torch.float32, device=device).reshape(1000, T, dim_u)
Y_train, _ = generate_observations(U_train, n_sensors=N, r=r2)
df2 = pd.read_csv("Data/diffusion_equation/u_test_Ns={}_T={}_sigma={}.csv".format(dim_u, T, sigma))
U_test = torch.tensor(df2.values, dtype=torch.float32, device=device).reshape(200, T, dim_u)
Y_test, _ = generate_observations(U_test, n_sensors=N, r=r2)

loader_train = torch.utils.data.DataLoader(TensorDataset(U_train, Y_train), batch_size=128,
                                           shuffle=True, drop_last=True)
loader_test = torch.utils.data.DataLoader(TensorDataset(U_test, Y_test), batch_size=128,
                                           shuffle=True, drop_last=True)
arch_params = {"Tx_units": 64, "Tx_layers": 6, "Ty_units": 64, "Ty_layers": 6,
               "A_net_units": 64, "A_net_layers": 6, "B_net_units": 64, "B_net_layers": 6}
model = Flow_based_Bayesian_Filter(arch_params, dim_u, dim_y, loader_train, loader_test, device)
model_dir = "FBF/diffusion_equation/r2={}/FBF_sensor={}_{}_{}_sigma={}".format(r2, N, dim_u, T, sigma)
model.load_model(model_dir, device)

dt = 1/T
s_min, s_max = -1.0, 1.0
s_grid = torch.linspace(s_min, s_max, dim_u+2)  # 包含边界
s_inner = s_grid[1:-1]
m_0 = -torch.sin(np.pi * s_inner)

N1 = 20
mmd_all_sims = torch.zeros((N1, T-1), device=device)
crps_all_sims = torch.zeros((N1, T-1), device=device)
rmse_list = []

gamma_param = 1/(2*2**2)
for idx_test in range(N1):
    print(idx_test)
    ensemble_size = 500
    measurement = Y_test[idx_test].to(device)

    SAD = SimpleNamespace()
    SAD.sigma = sigma
    SAD.dt = 1/100
    SAD.Nt = T

    x_ensemble_sample = model.calc_ensemble(ensemble_size, m_0,  measurement, SAD, device)# [99,500,50]
    x_rmse = torch.sqrt(((U_test[idx_test, 1:] - x_ensemble_sample.mean(1)) ** 2).mean())
    rmse_list.append(x_rmse)

    for k in range(T - 1):
        current_particles = x_ensemble_sample[k]  # [N_particles, dim_u], tensor
        current_truth = U_test[idx_test, k + 1, :]  # [dim_u], tensor
        current_truth_reshaped = current_truth.unsqueeze(0)  # [1, dim_u]

        # --- MMD ---
        mmd_value = calculate_mmd(current_particles, current_truth_reshaped, kernel_gamma=gamma_param)
        mmd_value = torch.as_tensor(mmd_value, device=device, dtype=torch.float32)
        mmd_all_sims[idx_test,k] = mmd_value
        # results_mmd.append(mmd_value)

        crps_components = []
        for d in range(dim_u):
            particles_d = current_particles[:, d]
            truth_d = current_truth[d]
            crps_d_value = calculate_crps(particles_d, truth_d)
            crps_components.append(crps_d_value)

        crps_value = torch.mean(torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32)
                                             for v in crps_components]))
        crps_all_sims[idx_test,k] = crps_value
        # results_crps.append(crps_value)
# --- 计算均值和标准差 ---
print(model_dir)
RMSE=torch.stack(rmse_list)

MMD = mmd_all_sims.mean(1)
CRPS = crps_all_sims.mean(1)
RMSE_mean = RMSE.mean()
RMSE_std = RMSE.std(unbiased=True)
print(MMD.shape)
print('RMSE mean:', RMSE_mean.item(), 'std:', RMSE_std.item())

MMD_mean = torch.mean(MMD)
MMD_std = MMD.std(unbiased=True)

CRPS_mean = CRPS.mean()
CRPS_std = CRPS.std(unbiased=True)

print('MMD mean:', MMD_mean.item(), 'std:', MMD_std.item())
print('CRPS mean:', CRPS_mean.item(), 'std:', CRPS_std.item())

x = np.linspace(-1.0, 1.0, dim_u + 2)[1:-1]
sensor_idx = torch.linspace(0, dim_u - 1, N).long()
x1 = x[sensor_idx]
def visualize_confidence_interval_spatial(T, true_state, x_ensemble_sample, x_P_plus):
    idx1 = int(T * 0.5)
    idx2 = int(T * 0.75)
    idx3 = max(0, T - 2)  # 最后一个有效索引
    time_points_to_plot = [int(T * 0.5), int(T * 0.75), int(T * 0.99)]
    Nx = true_state.shape[1]
    t1_mean = x_ensemble_sample[idx1].mean(0)
    t2_mean = x_ensemble_sample[idx2].mean(0)
    t3_mean = x_ensemble_sample[idx3].mean(0)

    t1_std = x_ensemble_sample[idx1].std(0)
    t2_std = x_ensemble_sample[idx2].std(0)
    t3_std = x_ensemble_sample[idx3].std(0)

    t1_lb = t1_mean - 3 * t1_std
    t1_ub = t1_mean + 3 * t1_std
    t2_lb = t2_mean - 3 * t2_std
    t2_ub = t2_mean + 3 * t2_std
    t3_lb = t3_mean - 3 * t3_std
    t3_ub = t3_mean + 3 * t3_std

    plt.style.use('default')
    fig = plt.figure(figsize=(16, 5))  # 横向画布更宽

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, true_state[idx1], label='Truth',color="orange")
    ax.plot(x, t1_mean, label='FBF')
    ax.set_xlabel(r'$t = 0.5$', fontsize=18)

    ax.fill_between(x, t1_lb, t1_ub, facecolor='deepskyblue', label=r'$\pm 3$ Std')
    ax.scatter(x1, true_state[idx1][sensor_idx], label="sensor", color="orange")
    ax.set_ylim(bottom=-4)
    ax.legend(fontsize=12,loc='upper right')


    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, true_state[idx2], label='Truth',color="orange")
    ax.plot(x, t2_mean, label='FBF')
    ax.set_xlabel(r'$t = 0.75$', fontsize=18)
    ax.fill_between(x, t2_lb, t2_ub, facecolor='deepskyblue', label=r'$\pm 3$ Std')
    ax.scatter(x1, true_state[idx2][sensor_idx], label="sensor", color="orange")
    ax.set_ylim(bottom=-4)
    ax.legend(fontsize=12,loc='upper right')

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, true_state[idx3], label='Truth',color="orange")
    ax.plot(x, t3_mean, label='FBF')
    ax.set_xlabel(r'$t = 1$', fontsize=18)
    ax.legend(fontsize=12)
    ax.fill_between(x, t3_lb, t3_ub, facecolor='deepskyblue', label=r'$\pm 3$ Std')
    ax.scatter(x1, true_state[idx3][sensor_idx], label="sensor", color="orange")
    ax.set_ylim(bottom=-4)
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()

# l = 0
# x_true_value = U_test[l, 1:]
# measurement = Y_test[l].to(device)
# SAD = SimpleNamespace()
# SAD.sigma = sigma
# SAD.dt = 1/100
# SAD.Nt = T
#
#
# def Calc_sample_covariance(x_ensemble_sample):
#     ensemble_size = x_ensemble_sample.shape[1]
#     x_mean = x_ensemble_sample.mean(axis=1, keepdims=True)
#     x_dev = x_ensemble_sample - x_mean
#     return 1 / (ensemble_size - 1) * np.matmul(x_dev.transpose(0, 2, 1), x_dev)
# x_ensemble_sample = model.calc_ensemble(ensemble_size, measurement, SAD, device).cpu().detach().numpy()
# print(x_ensemble_sample.shape)
# x_P_plus = Calc_sample_covariance(x_ensemble_sample)
# visualize_confidence_interval_spatial(100, x_true_value, x_ensemble_sample, x_P_plus)
#
