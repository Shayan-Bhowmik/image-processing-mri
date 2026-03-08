import torch
import torch.nn as nn
import torch.optim as optim

def get_training_components(model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, criterion, optimizer, device