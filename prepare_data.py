import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class Triple(Dataset):
    def __init__(self):
        super(Triple, self).__init__()
        self.entity_dic, self.relation_dic = self.load_texd()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def load_texd(self):
        raw_data = pd.read_csv('./fb15k/freebase_mtr100_mte100-train.txt', sep='\t', header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False, encoding='utf-8')
        head_count = Counter(raw_data['head'])
        tail_count = Counter(raw_data['tail'])
        relation_count = Counter(raw_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(idx, word) for idx, word in enumerate(entity_list)])
        relation_dic = dict([(idx, word) for idx, word in enumerate(relation_list)])
        return entity_dic, relation_dic
