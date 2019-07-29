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
        :return: pos_head_and_relation: [batch_size, embed_dim]
                 pos_tail: [batch_size, embed_dim]
                 neg_head_and_relation: [batch_size, embed_dim]
                 neg_tail: [batch_size, embed_dim]
        """
        pos_head_and_relation = self.entity_embedding(pos_head) + self.relation_embedding(pos_relation)
        pos_tail = self.entity_embedding(pos_tail)
        neg_head_and_relation = self.entity_embedding(neg_head) + self.relation_embedding(neg_relation)
        neg_tail = self.entity_embedding(neg_tail)
        return pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail


if __name__ == '__main__':
    dataset = Triple()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    transe = TranE(dataset.entity_num, dataset.relation_num)
    for batch_idx, (pos, neg) in enumerate(loader):
        break
