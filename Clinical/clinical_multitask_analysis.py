import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings

# 忽略一些警告让输出更干净
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # 文件路径 (请确保这些路径在你本地是正确的)
    METADATA_PATH = '/home/Guanjq/Work/OSCC-PathologyImageDataset/Data/all_metadata.json'
    SPLIT_PATH = '/home/Guanjq/Work/OSCC-PathologyImageDataset/Data/split_seed=2024.json'
    CLINICAL_PATH = '/home/Guanjq/Work/OSCC-PathologyImageDataset/Clinical/clinical_data_2024.csv'
    
    # 模型设置
    BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT" 
    MAX_LEN = 512  
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 5e-5
    SEED = 2026
    N_BOOTSTRAPS = 500  # Bootstrap 迭代次数
    
    # 任务定义: 任务名 -> 类别数
    TASKS = {
        "TD": 3,  # Tumor Differentiation (3分类)
        "CE": 2,
        "TI": 2,
        "PI": 2,
        "LNM": 2,
        "REC": 2,
    }

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# ==========================================
# 2. 数据处理 (Data Processing)
# ==========================================

class OSCCClinicalDataset(Dataset):
    def __init__(self, metadata, clinical_df, split_pids, tokenizer, max_len, is_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        
        self.samples = []
        
        meta_dict = {item['pid']: item for item in metadata['datainfo']}
        clinical_records = clinical_df.set_index('PID').to_dict('index')

        for pid in split_pids:
            if pid in meta_dict and pid in clinical_records:
                label_info = meta_dict[pid]
                feat_info = clinical_records[pid]
                
                # 收集所有标签
                labels = {task: label_info[task] for task in Config.TASKS.keys()}
                
                self.samples.append({
                    'pid': pid,
                    'features': feat_info,
                    'labels': labels
                })
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        features = item['features']
        labels = item['labels']
        
        # --- 文本构建与数据增强 ---
        text_parts = []
        for k, v in features.items():
            s_val = str(v).strip()
            if s_val == '/' or s_val.lower() == 'nan' or s_val == '':
                continue 
            text_parts.append(f"{k}: {s_val}")
        
        # Random Shuffle Columns (仅在训练集应用)
        if self.is_train:
            random.shuffle(text_parts)
        else:
            text_parts.sort() # 保证验证集顺序一致
            
        full_text = " ".join(text_parts)
        
        # --- Tokenization ---
        encoding = self.tokenizer.encode_plus(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        labels_tensor = {t: torch.tensor(l, dtype=torch.long) for t, l in labels.items()}

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels_tensor,
            'pid': item['pid']
        }

def load_data():
    print("Loading Data...")
    with open(Config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    with open(Config.SPLIT_PATH, 'r') as f:
        splits = json.load(f)
        
    clinical_df = pd.read_excel(Config.CLINICAL_PATH)

    # drop labels
    clinical_df = clinical_df.drop(columns=['[annotation] recurrence time', '[annotation] last followup time'])

    clinical_df.columns = [c.strip() for c in clinical_df.columns]
    if 'PID' not in clinical_df.columns:
        clinical_df.rename(columns={clinical_df.columns[0]: 'PID'}, inplace=True)
    
    clinical_df['PID'] = pd.to_numeric(clinical_df['PID'], errors='coerce')
    clinical_df = clinical_df.dropna(subset=['PID'])
    clinical_df['PID'] = clinical_df['PID'].astype(int)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    datasets = {}
    dataloaders = {}
    for phase in ['train', 'valid', 'test']:
        is_train = (phase == 'train')
        datasets[phase] = OSCCClinicalDataset(
            metadata=metadata,
            clinical_df=clinical_df,
            split_pids=splits[phase],
            tokenizer=tokenizer,
            max_len=Config.MAX_LEN,
            is_train=is_train
        )
        dataloaders[phase] = DataLoader(
            datasets[phase], 
            batch_size=Config.BATCH_SIZE, 
            shuffle=is_train, 
            num_workers=2
        )
        print(f"  {phase}: {len(datasets[phase])} samples")
    
    return dataloaders

# ==========================================
# 3. 模型定义 (Model Architecture)
# ==========================================

class ClinicalBertMultiTask(nn.Module):
    def __init__(self, task_config=Config.TASKS):
        super(ClinicalBertMultiTask, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        
        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_config.items():
            self.heads[task_name] = nn.Linear(self.bert.config.hidden_size, num_classes)
            
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        
        logits = {}
        for task_name, head in self.heads.items():
            logits[task_name] = head(output)
            
        return logits

# ==========================================
# 4. 统计与辅助函数 (Metrics & Helpers)
# ==========================================

def calculate_auc(y_true, y_score, num_classes):
    """ 计算单次 AUC """
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, y_score[:, 1])
        else:
            return roc_auc_score(y_true, y_score, multi_class='ovr')
    except ValueError:
        # 极少情况：如果所有样本都是同一类，AUC无法计算
        return 0.5

def bootstrap_auc(y_true, y_score, num_classes, n_bootstraps=500):
    """
    Bootstrap 计算 AUC 的 95% 置信区间
    """
    # 1. 计算原始 AUC
    original_auc = calculate_auc(y_true, y_score, num_classes)
    
    rng = np.random.RandomState(Config.SEED)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # 重采样索引 (replacement=True)
        indices = rng.randint(0, len(y_true), len(y_true))
        
        # 检查重采样后的标签是否包含至少两个类别，否则无法计算 AUC
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        score = calculate_auc(y_true[indices], y_score[indices], num_classes)
        bootstrapped_scores.append(score)
    
    if len(bootstrapped_scores) == 0:
        return original_auc, original_auc, original_auc

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # 95% CI 对应 2.5th 和 97.5th 分位数
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return original_auc, confidence_lower, confidence_upper

# ==========================================
# 5. 训练与评估逻辑
# ==========================================

def train_epoch(model, dataloader, optimizer, scheduler, active_tasks, device, criterion):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}
        
        optimizer.zero_grad()
        logits_dict = model(input_ids, attention_mask)
        
        loss = 0
        for task in active_tasks:
            task_loss = criterion(logits_dict[task], labels[task])
            loss += task_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, active_tasks, device, calc_ci=False):
    """
    calc_ci=True 时，返回详细字典 {task: {'auc': 0.8, 'low': 0.7, 'high': 0.9}}
    calc_ci=False 时，只返回字典 {task: 0.8}
    """
    model.eval()
    all_preds = {task: [] for task in active_tasks}
    all_targets = {task: [] for task in active_tasks}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            logits_dict = model(input_ids, attention_mask)
            
            for task in active_tasks:
                probs = torch.softmax(logits_dict[task], dim=1)
                all_preds[task].append(probs.cpu().numpy())
                all_targets[task].append(labels[task].cpu().numpy())
    
    results = {}
    for task in active_tasks:
        y_true = np.concatenate(all_targets[task])
        y_score = np.concatenate(all_preds[task])
        num_classes = Config.TASKS[task]
        
        if calc_ci:
            # 测试集计算 CI
            auc, low, high = bootstrap_auc(y_true, y_score, num_classes, n_bootstraps=Config.N_BOOTSTRAPS)
            results[task] = {'auc': auc, 'low': low, 'high': high}
        else:
            # 验证集只计算 AUC 均值
            auc = calculate_auc(y_true, y_score, num_classes)
            results[task] = auc
        
    return results

def run_training_session(dataloaders, active_tasks, session_name):
    device = Config.DEVICE
    print(f"\n>>> Starting Session: {session_name}")
    print(f"Tasks involved: {active_tasks}")
    
    model = ClinicalBertMultiTask(task_config=Config.TASKS).to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = len(dataloaders['train']) * Config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc_avg = 0
    # 我们只在训练结束时对最佳模型进行 Test 评估，或者这里简化为最后一次
    # 为简单起见，这里保存最佳验证集的模型状态，最后再测一次 Test
    best_model_state = None
    
    for epoch in range(Config.EPOCHS):
        loss = train_epoch(model, dataloaders['train'], optimizer, scheduler, active_tasks, device, criterion)
        
        # 验证集 (不计算 CI)
        val_aucs = evaluate(model, dataloaders['valid'], active_tasks, device, calc_ci=False)
        avg_val_auc = np.mean(list(val_aucs.values()))
        
        auc_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_aucs.items()])
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {loss:.4f} | Avg Val AUC: {avg_val_auc:.4f} | Details: {auc_str}")
        
        if avg_val_auc > best_val_auc_avg:
            best_val_auc_avg = avg_val_auc
            best_model_state = model.state_dict()
            
    # 加载最佳验证集模型进行测试
    print(f"\nEvaluating {session_name} on Test Set (with 95% CI)...")
    model.load_state_dict(best_model_state)
    
    # 测试集 (计算 CI)
    test_results = evaluate(model, dataloaders['test'], active_tasks, device, calc_ci=True)
    
    for t, res in test_results.items():
        print(f"  Task {t}: {res['auc']:.4f} (95% CI: {res['low']:.4f}-{res['high']:.4f})")
        
    return test_results

# ==========================================
# 6. 主程序 (6+1 Execution Flow)
# ==========================================

if __name__ == "__main__":
    dataloaders = load_data()
    
    # 用于汇总所有结果
    # 格式化为字符串: "0.8521 (0.8300-0.8700)"
    final_summary = {
        'Task': list(Config.TASKS.keys()),
        'Single_Task_Result': [""] * len(Config.TASKS),
        'Multi_Task_Result': [""] * len(Config.TASKS)
    }
    
    # --- Part 1: Single Task ---
    print("\n" + "="*60)
    print(f"PHASE 1: SINGLE TASK LEARNING (Bootstraps={Config.N_BOOTSTRAPS})")
    print("="*60)
    
    task_list = list(Config.TASKS.keys())
    
    for i, task in enumerate(task_list):
        exp_name = f"Exp {i+1}/7 (Single Task - {task})"
        result = run_training_session(dataloaders, active_tasks=[task], session_name=exp_name)
        
        # 提取结果并格式化
        r = result[task] # {'auc':..., 'low':..., 'high':...}
        res_str = f"{r['auc']:.4f} ({r['low']:.4f}-{r['high']:.4f})"
        
        idx = final_summary['Task'].index(task)
        final_summary['Single_Task_Result'][idx] = res_str

    # --- Part 2: Multi Task ---
    print("\n" + "="*60)
    print("PHASE 2: MULTI TASK LEARNING")
    print("="*60)
    
    exp_name = "Exp 7/7 (Multi Task - All)"
    result = run_training_session(dataloaders, active_tasks=task_list, session_name=exp_name)
    
    for task, r in result.items():
        res_str = f"{r['auc']:.4f} ({r['low']:.4f}-{r['high']:.4f})"
        idx = final_summary['Task'].index(task)
        final_summary['Multi_Task_Result'][idx] = res_str
        
    # --- Summary ---
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (AUC & 95% CI)")
    print("="*80)
    
    df_res = pd.DataFrame(final_summary)
    # 设置列对齐使得打印更好看
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(df_res.to_string(index=False))
    
    # 保存结果
    save_path = "/home/Guanjq/Work/OSCC-PathologyImageDataset/Clinical/oscc_clinicalbert_results_with_ci.csv"
    df_res.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")