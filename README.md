# Pytorch Implementation of TransE

Pytorch version: 1.1.0

**Paper：**
- [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)

**Dataset：**
- [FB15k](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz)

To evaluate, we do tail prediction on the test set, and this TransE model reaches hits@10 of **34.5%**, which is similar to the raw performance mentioned in the paper.