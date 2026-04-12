import argparse

def get_args():
    parser = argparse.ArgumentParser(description='LMAGNN: Logic Modeling GNN for Knowledge Graph Completion')
    
    # Dataset
    parser.add_argument('--data_path', type=str, default='./data/WN18RR/', help='Path to the dataset')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--fact_ratio', type=float, default=0.96, help='Ratio of facts in KG')
    
    # Model Architecture
    parser.add_argument('--n_layer', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension size')
    parser.add_argument('--attn_dim', type=int, default=32, help='Attention dimension size')
    parser.add_argument('--topk', type=int, default=1000, help='Top-k nodes to sample per layer')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for Gumbel-Softmax')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--lamb', type=float, default=0.0001, help='Weight decay (lambda)')
    parser.add_argument('--decay_rate', type=float, default=0.994, help='Exponential LR decay rate')
    parser.add_argument('--n_batch', type=int, default=50, help='Batch size for training')
    parser.add_argument('--n_tbatch', type=int, default=50, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    # Misc
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save models')
    
    args = parser.parse_args()
    args.n_node_topk = [args.topk] * args.n_layer
    return args