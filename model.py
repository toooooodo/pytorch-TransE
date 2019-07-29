import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prepare_data import Triple
import math


class TranE(nn.Module):
    def __init__(self, entity_num, relation_num, dim=50):
        super(TranE, self).__init__()
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, dim).uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim)))
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, dim).uniform_(-6 / math.sqrt(dim), 6 / math.sqrt(dim)))
        # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1).view(-1, 1)
        _, relation_norm = torch.broadcast_tensors(self.relation_embedding.weight.data, relation_norm)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm

    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        """
        :param pos_head: [batch_size]
        :param pos_relation: [batch_size]
        :param pos_tail: [batch_size]
        :param neg_head: [batch_size]
        :param neg_relation: [batch_size]
        :param neg_tail: [batch_size]
        :return: positive: [batch_size, embed_dim]
                 negative: [batch_size, embed_dim]
        """
        pos_h, pos_t = self.entity_embedding(pos_head), self.entity_embedding(pos_tail)
        pos_r = self.relation_embedding(pos_relation)
        neg_h, neg_t = self.entity_embedding(neg_tail), self.entity_embedding(neg_tail)
        neg_r = self.relation_embedding(neg_relation)
        positive = pos_h + pos_r - pos_t
        negative = neg_h + neg_r - neg_t
        return positive, negative


if __name__ == '__main__':
    dataset = Triple()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    transe = TranE(dataset.entity_num, dataset.relation_num)
    for batch_idx, (pos, neg) in enumerate(loader):
        break
