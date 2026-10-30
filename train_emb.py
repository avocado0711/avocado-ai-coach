import warnings
warnings.filterwarnings("ignore")
import os
# 国内镜像加速huggingface下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
import json
import random

# -------------------------- 配置
random.seed(42)
base_model = "BAAI/bge-small-zh-v1.5"
batch_size = 8
epochs = 8
output_dir = r"D:\26572\dunk\fit-sport-emb"
# --------------------------

# 读取原始全部数据
all_data = []
with open(r"D:\26572\dunk\train_data.json1", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        all_data.append(json.loads(line))

# 划分 330训练 / 89验证
random.shuffle(all_data)
train_data = all_data[:330]
val_data = all_data[330:]

# 组装训练样本
train_examples = []
for item in train_data:
    train_examples.append(InputExample(texts=[item["query"], item["positive"], item["negative"]]))

# 组装验证集
val_queries = []
val_positives = []
val_negatives = []
for item in val_data:
    val_queries.append(item["query"])
    val_positives.append(item["positive"])
    val_negatives.append(item["negative"])

# 加载基础模型
model = SentenceTransformer(base_model)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.TripletLoss(model)

evaluator = TripletEvaluator(val_queries, val_positives, val_negatives)
print("训练前验证集指标：")
evaluator(model)

warmup_steps = int(len(train_dataloader) * epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path=output_dir,
    save_best_model=True
)

print(f"训练完成，模型保存在：{output_dir}")