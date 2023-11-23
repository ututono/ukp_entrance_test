from src.utils.global_variables import TEST_FILENAME, TRAIN_FILENAME, DEV_FILENAME


class DataProcessor():
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def load_raw_data(self):
        raise NotImplementedError

