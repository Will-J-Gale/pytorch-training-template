import math
from random import shuffle
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

class DataGenerator:
    def __init__(self, dataset, batch_size=32, device=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.input_batch = []
        self.output_batch = []
        self.dataset_indexes = np.arange(len(dataset))
        self.shuffle()

    def load_data(self, data):
        #@TODO Return train_X, train_Y for single part of batch
        raise NotImplementedError

    def on_data_loaded(self, future):
        x, y = future.result()
        self.input_batch.append(x)
        self.input_batch.append(y)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        batch_indexes = self.dataset_indexes[start:end]
        self.input_batch = []
        self.output_batch = []
    
        with ThreadPoolExecutor(max_workers=8) as executor:
            for dataset_index in batch_indexes:
                job = executor.submit(self.load_data, self.dataset[dataset_index])
                job.add_done_callback(self.on_data_loaded)

        train_X = np.array(self.input_batch, dtype=np.float32)
        train_Y = np.array(self.output_batch, dtype=np.float32)

        train_X = torch.from_numpy(train_X).to(self.device)
        train_Y = torch.from_numpy(train_Y).to(self.device)

        return train_X, train_Y 
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
    
    def shuffle(self):
        shuffle(self.dataset_indexes)