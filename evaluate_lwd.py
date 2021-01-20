import torch
import os
import argparse
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from env import MaximumIndependentSetEnv
from data.graph_dataset import get_er_15_20_dataset
from torch.utils.data import DataLoader
import dgl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    help="directory of the graphs to evaluate",
    type=str
    )
parser.add_argument(
    "--device",
    help="id of gpu device to use",
    type=str
    )

args = parser.parse_args()

device = torch.device(args.device)

data_dir = args.data_dir
model_name = "lwd.pt"
model_path = os.path.join('saved_models', model_name)


# env
hamming_reward_coef = 0.1

# actor critic
num_layers = 4
input_dim = 2
output_dim = 3
hidden_dim = 128

# optimization
init_lr = 1e-4
max_epi_t = 32
max_rollout_t = 32
max_update_t = 20000

# ppo
gamma = 1.0
clip_value = 0.2
optim_num_samples = 4
critic_loss_coef = 0.5
reg_coef = 0.1
max_grad_norm = 0.5

# logging
vali_freq = 50
log_freq = 10

# dataset specific
dataset = "synthetic"
graph_type = "er"
min_num_nodes = 15
max_num_nodes = 20

# main
rollout_batch_size = 32
eval_batch_size = 1000
optim_batch_size = 16
init_anneal_ratio = 1.0
max_anneal_t = - 1
anneal_base = 0.
train_num_samples = 2
eval_num_samples = 10


# initial values
best_vali_sol = -1e5

# generate and save datasets
num_eval_graphs = 1000




model = ActorCritic(
    actor_class = PolicyGraphConvNet,
    critic_class = ValueGraphConvNet,
    max_num_nodes = max_num_nodes,
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    device = device
    )

model.load_state_dict(torch.load(model_path, map_location={'cuda:0': args.device}))
parametri = list(model.parameters())
shapes = [par.shape for par in parametri]


env = MaximumIndependentSetEnv(
    max_epi_t = max_epi_t,
    max_num_nodes = max_num_nodes,
    hamming_reward_coef = hamming_reward_coef,
    device = device
    )



def evaluate(mode, actor_critic):
    eval = True
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_sol = 0.0
    for g in data_loaders[mode]:
        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(g, num_samples = eval_num_samples)
        #g = g.to(device) dice che g è un batchedgraph e non ha l'attributo to
        while True:
            with torch.no_grad():
                prob = actor_critic.act(ob, g)

            ob, reward, done, info = env.step(prob)
            if torch.all(done).item():
                cum_eval_sol += info['sol'].max(dim = 1)[0].sum().cpu()
                cum_cnt += g.batch_size
                break


    avg_eval_sol = cum_eval_sol / cum_cnt

    return avg_eval_sol

for directory in os.listdir(data_dir):
    print('Evaluating {}'.format(directory))
    datasets = {
        "vali": get_er_15_20_dataset("vali", os.path.join(data_dir, directory))
        }

    # construct data loaders
    def collate_fn(graphs):
        return dgl.batch(graphs)

    data_loaders = {

        "vali": DataLoader(
            datasets["vali"],
            batch_size = eval_batch_size,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = 0
            )
            }
    media = evaluate('vali', model)
    print(media)
