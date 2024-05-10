import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

class MyModel(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super(MyModel, self).__init__()
        self.esm2 = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    def forward(self, input_ids):
        pooled_output = self.esm2(input_ids).pooler_output
        print(self.esm2(input_ids))
        # print(pooled_output)
        # pooled_output = self.bert_model(input_ids)
        return pooled_output


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", do_lower_case=False)
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=60, truncation=True)
        print(inputs)
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)

df = pd.read_csv('./esm2data/realdata.csv')
print('一共有{}条数据'.format(len(df)))
df.info()
use_df = df[:]
use_df.head(10)
features = df['feature'].tolist()
labels = df['label'].tolist()

df1 = pd.read_csv('./esm2data/fakedata.csv')
print('一共有{}条数据'.format(len(df1)))
df1.info()
use_df = df1[:]
use_df.head(10)
features1 = df1['feature'].tolist()
labels1 = df1['label'].tolist()

real_dataset = MyDataSet(features, labels)
realloader = Data.DataLoader(real_dataset, batch_size=1, shuffle=False)
fake_dataset = MyDataSet(features1, labels1)
fakeloader = Data.DataLoader(fake_dataset, batch_size=1, shuffle=False)

model = MyModel()

for name, parameters in model.named_parameters():
    print(name, ';', parameters.size())

print(model)

for input_ids, _, _ in realloader:
    real_input = input_ids
    print(real_input.shape)
    real_pred = model(input_ids)
    print(real_pred)

for input_ids, _, _  in fakeloader:
    fake_input = input_ids
    fake_pred = model(input_ids)
    fake_pred = model(input_ids)

# 从文件或其他数据源获取蛋白质结构的坐标数据，然后将其存储为NumPy数组
# 假设protein1_coords和protein2_coords是两个包含坐标数据的NumPy数组

# 检查原子数量是否相同
if len(real_input) != len(fake_input):
    raise ValueError("原子数量不匹配")
# 计算每对原子之间的坐标差的平方
diff_squared = (real_input - fake_input) ** 2
# 计算坐标差的平方的和
sum_of_squared_diff = np.sum(diff_squared)
# 计算RMSD
rmsd_before = np.sqrt(sum_of_squared_diff / len(real_pred))
print("扩充前RMSD:", rmsd_before)

if len(real_pred) != len(fake_pred):
    raise ValueError("原子数量不匹配")
# 计算每对原子之间的坐标差的平方
diff_squared = (real_pred - fake_pred) ** 2
# 计算坐标差的平方的和
sum_of_squared_diff = np.sum(diff_squared)
# 计算RMSD
rmsd_after = np.sqrt(sum_of_squared_diff / len(real_pred))

print("扩充后RMSD:", rmsd_after)
