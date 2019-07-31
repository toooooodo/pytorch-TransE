import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from prepare_data import TrainSet, TestSet
import math


class TranE(nn.Module):
    def __init__(self, entity_num, relation_num, device, dim=50, d_norm=2, gamma=1):
        """
        :param entity_num: number of entities
        :param relation_num: number of relations
        :param dim: embedding dim
        :param device:
        :param d_norm: measure d(h+l, t), either L1-norm or L2-norm
        :param gamma: margin hyperparameter
        """
        super(TranE, self).__init__()
        self.dim = dim
        self.d_norm = d_norm
        self.device = device
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(relation_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)),
            freeze=False)
        # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm

    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        """
        :param pos_head: [batch_size]
        :param pos_relation: [batch_size]
        :param pos_tail: [batch_size]
        :param neg_head: [batch_size]
        :param neg_relation: [batch_size]
        :param neg_tail: [batch_size]
        :return: triples loss
        """
        pos_dis = self.entity_embedding(pos_head) + self.relation_embedding(pos_relation) - self.entity_embedding(
            pos_tail)
        neg_dis = self.entity_embedding(neg_head) + self.relation_embedding(neg_relation) - self.entity_embedding(
            neg_tail)
        # return pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail
        return self.calculate_loss(pos_dis, neg_dis).requires_grad_()

    def calculate_loss(self, pos_dis, neg_dis):
        """
        :param pos_dis: [batch_size, embed_dim]
        :param neg_dis: [batch_size, embed_dim]
        :return: triples loss: [batch_size]
        """
        distance_diff = self.gamma + torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis, p=self.d_norm,
                                                                                            dim=1)
        return torch.sum(F.relu(distance_diff))

    def tail_predict(self, head, relation, tail, k=10):
        """
        to do tail prediction hits@k
        :param head: [batch_size]
        :param relation: [batch_size]
        :param tail: [batch_size]
        :param k: hits@k
        :return:
        """
        # head: [batch_size]
        # h_and_r: [batch_size, embed_size] => [batch_size, 1, embed_size] => [batch_size, N, embed_size]
        h_and_r = self.entity_embedding(head) + self.relation_embedding(relation)
        h_and_r = torch.unsqueeze(h_and_r, dim=1)
        h_and_r = h_and_r.expand(h_and_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch_size, N, embed_size]
        embed_tail = self.entity_embedding.weight.data.expand(h_and_r.shape[0], self.entity_num, self.dim)
        # indices: [batch_size, k]
        values, indices = torch.topk(torch.norm(h_and_r - embed_tail, dim=2), k, dim=1, largest=False)
        # tail: [batch_size] => [batch_size, 1]
        tail = tail.view(-1, 1)
        return torch.sum(torch.eq(indices, tail)).item()


if __name__ == '__main__':
    train_data_set = TrainSet()
    test_data_set = TestSet()
    test_data_set.convert_word_to_index(train_data_set.entity_to_index, train_data_set.relation_to_index,
                                        test_data_set.raw_data)
    train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_set, batch_size=32, shuffle=True)
    for batch_idx, data in enumerate(test_loader):
        print(data.shape)
        break
