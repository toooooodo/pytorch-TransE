import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from model import TranE
from prepare_data import TrainSet, TestSet

device = torch.device('cuda')
embed_dim = 50
num_epochs = 200
batch_size = 32
lr = 1e-2
momentum = 0
gamma = 1
d_norm = 2
top_k = 10


def main():
    train_dataset = TrainSet()
    test_dataset = TestSet()
    test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                       test_dataset.raw_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    transe = TranE(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim, d_norm=d_norm,
                   gamma=gamma).to(device)
    optimizer = optim.SGD(transe.parameters(), lr=lr, momentum=momentum)
    for epoch in range(num_epochs):

        # e <= e / ||e||
        entity_norm = torch.norm(transe.entity_embedding.weight.data, dim=1, keepdim=True)
        transe.entity_embedding.weight.data = transe.entity_embedding.weight.data / entity_norm
        total_loss = 0
        for batch_idx, (pos, neg) in enumerate(train_loader):
            pos, neg = pos.to(device), neg.to(device)
            # pos: [batch_size, 3] => [3, batch_size]
            pos = torch.transpose(pos, 0, 1)
            # pos_head, pos_relation, pos_tail: [batch_size]
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
            neg = torch.transpose(neg, 0, 1)
            # neg_head, neg_relation, neg_tail: [batch_size]
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            loss = transe(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}, loss = {total_loss/train_dataset.__len__()}")
        if epoch % 20 == 0:
            corrct_test = 0
            for batch_idx, data in enumerate(test_loader):
                # data: [1, 3]
                corrct_test += transe.tail_predict(data[0, 0], data[0, 1], data[0, 2], k=top_k)
            print(f"epoch {epoch+1}, test accuracy {corrct_test/test_dataset.__len__()}")


if __name__ == '__main__':
    main()
