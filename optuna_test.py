import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna 
import os
from optuna.trial import TrialState

import optuna.visualization.matplotlib as optuna_vis
import matplotlib.pyplot as plt

import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")
print(f"Using device: {device}")

DEVICE = device
BATCHSIZE = 32
BATCHSIZE_VAL = 500
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 300
N_VALID_EXAMPLES = BATCHSIZE_VAL * 5

def define_model(trial):
    # optimizing number of layers and number of base_filters, drop out ratio
    n_layers = trial.suggest_int("n_layers", 1, 3)
    drop_factor = trial.suggest_float("drop_factor", 0.0, 0.5)
    layers = []
    bf = trial.suggest_categorical("bf", [128, 256, 512])
    input_dims = 28*28
    output_dims = 10 
    layers.append(nn.Flatten())
    for i in range(1, n_layers):
        bf = bf//i
        layers.append(nn.Linear(input_dims, bf))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        input_dims = bf
    layers.append(nn.Linear(input_dims, output_dims))

    return nn.Sequential(*layers)
    


# ----- Data Loading (Torchvision MNIST - works for Kaggle or local) -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(trial):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=500,
        shuffle=True,
    )

    return train_loader, valid_loader

def getMNIST(transform):
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE_VAL, shuffle=False)

    return train_loader, test_loader

# ----- Training and Evaluation -----

def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist(trial)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE_VAL >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study           = optuna.create_study(direction="maximize", study_name="test-mnist-optuna", storage= f"sqlite:///{DIR}/test_mnist_optuna.db")
    study.optimize(objective, n_trials=50, timeout=600)
    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna_vis.plot_param_importances(
        study, 
        target=lambda t: t.duration.total_seconds(), 
        target_name="duration"
    )
    # Show the plot
    plt.show()

    # Save the plot as PNG (or other format)
    fig.savefig("param_importance_duration.png", dpi=300, bbox_inches='tight')
    print("Parameter importance plot saved as param_importance_duration.png")

    

