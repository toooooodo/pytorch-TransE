import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from model import TranE
from prepare_data import Triple

device = torch.device('cuda')
embed_dim = 50
num_epochs = 20
lr = 1e-2
momentum = 0
gamma = 1
d_norm = 2


def main():
    dataset = Triple()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    transe = TranE(dataset.entity_num, dataset.relation_num).to(device)
    optimizer = optim.SGD(transe.parameters(), lr=lr, momentum=momentum)
    for epoch in range(num_epochs):
        # e <= e / ||e||
        entity_norm = torch.norm(transe.entity_embedding.weight.data, dim=1).view(-1, 1)
        _, entity_norm = torch.broadcast_tensors(transe.entity_embedding.weight.data, entity_norm)
        transe.entity_embedding.weight.data = transe.entity_embedding.weight.data / entity_norm
        for batch_idx, (pos, neg) in enumerate(loader):
            pos, neg = pos.to(device), neg.to(device)
            # pos: [batch_size, 3] => [3, batch_size]
            pos = torch.transpose(pos, 0, 1)
            # pos_head, pos_relation, pos_tail: [batch_size]
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
            neg = torch.transpose(neg, 0, 1)
            # neg_head, neg_relation, neg_tail: [batch_size]
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail = transe(pos_head, pos_relation, pos_tail,
                                                                                      neg_head, neg_relation, neg_tail)
            loss = torch.max(torch.Tensor([0]).to(device),
                             gamma + torch.dist(pos_head_and_relation, pos_tail) - torch.dist(neg_head_and_relation,
                                                                                              neg_tail))
            print(loss)
            break
        break


if __name__ == '__main__':
    main()
