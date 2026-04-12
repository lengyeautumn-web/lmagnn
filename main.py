import torch
import numpy as np
from tqdm import tqdm
from config import get_args
from utils.data_loader import DataLoader
from utils.metrics import cal_ranks, cal_performance
from models.lmagnn import LMAGNN_LogicModel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import logging

class Engine:
    def __init__(self, args):
        self.args = args
        self.loader = DataLoader(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LMAGNN_LogicModel(args, self.loader).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)

    def train(self):
        best_mrr = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            n_batch = (self.loader.n_train + self.args.n_batch - 1) // self.args.n_batch
            
            for i in tqdm(range(n_batch), desc=f"Epoch {epoch}"):
                start, end = i * self.args.n_batch, min(self.loader.n_train, (i + 1) * self.args.n_batch)
                batch = self.loader.get_batch(np.arange(start, end))
                
                self.optimizer.zero_grad()
                scores = self.model(batch[:, 0], batch[:, 1])
                pos_scores = scores[torch.arange(len(batch)).to(self.device), torch.LongTensor(batch[:, 2]).to(self.device)]
                loss = torch.sum(- pos_scores + torch.logsumexp(scores, dim=1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            self.loader.shuffle_train()
            
            # Validation
            mrr, h1, h3, h10 = self.evaluate('valid')
            logging.info(f"Epoch {epoch} | Loss: {total_loss:.2f} | Val MRR: {mrr:.4f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                tmrr, th1, th3, th10 = self.evaluate('test')
                print(f"New Best! Test MRR: {tmrr:.4f}")
                torch.save(self.model.state_dict(), f"{self.args.save_path}best_model.pth")

    def evaluate(self, data_type='valid'):
        self.model.eval()
        n_data = self.loader.n_valid if data_type == 'valid' else self.loader.n_test
        batch_size = self.args.n_tbatch
        all_ranks = []
        
        for i in tqdm(range((n_data + batch_size - 1) // batch_size), desc=f"Eval {data_type}"):
            start, end = i * batch_size, min(n_data, (i + 1) * batch_size)
            subs, rels, objs = self.loader.get_batch(np.arange(start, end), data=data_type)
            with torch.no_grad():
                scores = self.model(subs, rels, mode=data_type).cpu().numpy()
            
            filters = np.zeros((len(subs), self.loader.n_ent))
            for j in range(len(subs)):
                filters[j, list(self.loader.filters[(subs[j], rels[j])])] = 1
            all_ranks += cal_ranks(scores, objs, filters)
        
        return cal_performance(all_ranks)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    engine = Engine(args)
    engine.train()