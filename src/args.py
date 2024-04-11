import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='data/few-shot') # few-shot / train_test_256
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--num_class', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    return parser