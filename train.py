from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import flwr as fl
from flwr.common import Metrics
import random
import torch.nn as nn
from typing import Union, Optional
DEVICE = torch.device("cpu")
from tqdm import tqdm
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate graph impact on stance predicition")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use during training.")
    parser.add_argument("--num_clients", type=int, default=40, help="The number of clients to use during federated learning")
    parser.add_argument("--rounds", type=int, default=10, help="The number of rounds to do federated.")
    parser.add_argument("--hidden_size", type=int, default=64, help="The hidden size of the model")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="The rng seed")
    args = parser.parse_args()
    return args

class RegressionData(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.x = df[['Periods','appliances']].values

    def __len__(self):
        return (len(self.x) // 96) - 8

    def __getitem__(self, idx):
        if idx > 358:
            raise Exception('Unbounded window')
        return torch.tensor(self.x[idx*96:(idx+7)*(96)], dtype=torch.float32), torch.tensor(self.x[(idx+7)*96:(idx+8)*96], dtype=torch.float32)

class MyRNN(nn.Module):
    def __init__(self, hid_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hid_size
        self.rnn = nn.LSTM(input_size=2, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)


    def forward(self, inputs, hidden_state, cell):
        outputs = []
        output, (hidden_states, cell) = self.rnn(inputs, (hidden_state, cell))
        output = self.fc1(output)
        output = nn.functional.relu(output)
        output = self.fc2(output)
        return output, hidden_states, cell

def train(net, train_loader):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    net.train()
    val_losses = []
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        for x, Y in train_loader:
            Y = Y.to(DEVICE)
            x = x.to(DEVICE)
            hidden = torch.zeros((num_layers, batch_size, net.hidden_size)).to(DEVICE)
            cell = torch.zeros((num_layers, batch_size, net.hidden_size)).to(DEVICE)
            outs, hidden, cell = net(x, hidden, cell)
            output, _, _ = net(Y, hidden, cell)
            z = torch.cat([outs[:, -1], output[:, :-1].squeeze()], dim=1)
            loss = criterion(Y[:, :, 1], z)
            val_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #print(f'{loss.item()}\r')
            optimizer.zero_grad()
        print(f'{total_loss / len(train_loader)}')
    return val_losses


def test(net, dataset, num_layers):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    net.eval()
    val_losses = []
    outputs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, Y = dataset[i]
            hidden = torch.zeros((num_layers, net.hidden_size)).to(DEVICE)
            cell = torch.zeros((num_layers, net.hidden_size)).to(DEVICE)
            outs, hidden, cell = net(x, hidden, cell)
            z = [outs[-1].squeeze().item()]
            for i in range(95):
                o, hidden, cell = net(torch.tensor([[i+1, z[-1]]]), hidden, cell)
                z.append(o.item())
            loss = criterion(Y[:, 1], torch.tensor(z))
            val_losses.append(loss.item())
            outputs += z
    return val_losses, outputs


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, dataset):
        self.net = net
        self.dataset = dataset
        self.losses = []

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss = train(self.net, self.dataset)
        self.losses = loss
        return get_parameters(self.net), len(self.dataset), {'loss': loss}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        return 0.0, 1, {}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    # Load model
    cid = int(cid) + 1
    net = MyRNN(hid_size=hidden_size).to(DEVICE)
    d = RegressionData(f'./data/Consumer{cid}.csv')
    train_loader = DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, train_loader)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ):
        """Aggregate model weights using weighted average and store checkpoint"""
        net = MyRNN(hid_size=64).to(DEVICE)

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    global batch_size
    global hidden_size
    global num_layers
    global epochs

    num_layers = 2
    hidden_size = args.hidden_size
    epochs = args.epochs
    batch_size = args.batch_size
    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=40,  # Wait until all 10 clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

