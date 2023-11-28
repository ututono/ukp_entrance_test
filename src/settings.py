import argparse
from src.utils_.global_variables import SEED

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model for a simple sequence tagger.')

    # <--- Arguments Definition --->
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Loss function')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load model from checkpoint')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to train/evaluate on')



    # <--- End of Arguments Definition --->

    args = parser.parse_args()
    return args






