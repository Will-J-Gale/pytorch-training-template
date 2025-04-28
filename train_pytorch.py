import os

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dataset_generator import DataGenerator

CHECKPOINT_PATH = "checkpoint.pt"

def save_checkpoint(model, optimizer, loss, epoch):
    global CHECKPOINT_PATH
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, device):
    global CHECKPOINT_PATH
    checkpoint = torch.load(CHECKPOINT_PATH,  map_location=torch.device(device), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]

training_data_path = "<PATH_TO_TRAINING_DATASET>" #@TODO Set path to training data
test_data_filepath = "<PATH_TO_TEST_DATASET>" #@TODO Set path to test data
device = torch.device("cuda:0")
weights_save_dir = "weights"
model_name = "<MODEL_NAME>" #@TODO Set model name
batch_size = 8
start_epoch = 0
epochs = 10
training_data = None #@TODO Load training data here
test_data = None #@TODO Load test data here
training_data_generator = DataGenerator(training_data, batch_size, device)
test_data_generator = DataGenerator(test_data, batch_size, device)
num_batches = len(training_data_generator) // batch_size
writer = SummaryWriter()

model = None #@TODO Create model here
model.compile()
loss_function = None #@TODO Create loss function here
optimizer = Adam(model.parameters(), lr=1e-4)

os.makedirs(weights_save_dir, exist_ok=True)

if(os.path.exists(CHECKPOINT_PATH)):
    start_epoch, loss = load_checkpoint(model, optimizer, device)

try:
    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        losses = []
        training_data_generator.shuffle()
        model.train()

        with tqdm(
            total=len(training_data_generator),
            bar_format="{n_fmt}/{total_fmt} {percentage:3.0f}%|{bar:20} | ETA: {elapsed}<{remaining} - Loss: {postfix}",
            postfix=0) as t:

            for i, (train_X, train_Y) in enumerate(training_data_generator):
                global_step = i + (epoch * len(training_data_generator))
                optimizer.zero_grad()

                outputs = model(train_X)
                loss = loss_function(outputs, train_Y)
                loss.backward()
                optimizer.step()
                
                loss_value = loss.item()
                losses.append(loss_value)

                writer.add_scalar("training_loss", loss_value, global_step)
                t.postfix = np.average(losses)
                t.update()

        model.eval()

        with torch.no_grad():
            test_losses = []
            for test_X, test_Y in tqdm(test_data_generator):
                outputs = model(test_X)

                loss = loss_function(outputs, test_Y)
                test_losses.append(loss.item())

            test_loss = np.average(test_losses)
            print(f"Val loss: {test_loss}")
        
        model_filename = f"{model_name}_{epoch}_{test_loss:.5f}"
        model.save(weights_save_dir, model_filename)
        save_checkpoint(model, optimizer, loss_value, epoch)

except KeyboardInterrupt as e:
    pass