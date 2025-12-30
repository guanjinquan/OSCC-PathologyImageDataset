import os
import json
import random
from collections import Counter


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/..")
    tasks =  ['TD', 'CE', 'TI', 'REC', 'PI', 'LNM']

    seed = 2024
    random.seed(seed)

    with open("./Data/all_metadata.json", 'r') as f:
        pathology_info = json.load(f)['datainfo']
        
    # collect the pids with label in each task
    pids = {}
    legal_pids = []
    pid_to_labels = {}
    for item in pathology_info:
        pid = item['pid']
        labels = [item[task] for task in tasks]
        if -1 in labels:
            continue
        pid_to_labels[pid] = {task: label for task, label in zip(tasks, labels)}
        legal_pids.append(pid)
        if tuple(labels) not in pids:
            pids[tuple(labels)] = [pid]
        else:
            pids[tuple(labels)].append(pid)

    print("Total number of legal pids: ", len(legal_pids))

    # compute the number of each class in each task
    task_cls_count = {}
    for task in tasks:
        task_cls_count[task] = Counter([item[task] for item in pathology_info if item['pid'] in legal_pids])

    # split the dataset
    start = 0
    while True:
        start += 1
        print("Now : ", start)
        train, valid, test = [], [], []
        for key, value in pids.items():
            random.shuffle(value)
            train_len = int(len(value) * 0.7)
            if random.random() < 0.5:
                valid_len = int(len(value) * 0.15)
                test_len = len(value) - train_len - valid_len
            else:
                test_len = int(len(value) * 0.15)
                valid_len = len(value) - train_len - test_len
            train += value[:train_len]
            valid += value[train_len:train_len+valid_len]
            test += value[train_len+valid_len:]

        # select the pid with label in major set in the set
        def select_major(data_pids):
            for pid in data_pids:
                flag = True
                for task in tasks:
                    if task_cls_count[task][pid_to_labels[pid][task]] < 200:
                        flag = False
                        break
                if flag:
                    return pid
            raise Exception("No major pid found")
            
        # limit the length of valid and test to 200
        for val in [valid, test]:
            if len(val) > 200:
                while len(val) > 200:
                    pid = select_major(val)
                    val.remove(pid)
                    train.append(pid)
            else:
                while len(val) < 200:
                    pid = select_major(train)
                    val.append(pid)
                    train.remove(pid)

        # check the distribution of each class in each mode [2 promises]
        flag = True
        task_mode_counter = {}
        for task in tasks:
            train_selected = [item['pid'] for item in pathology_info if item['pid'] in train]
            valid_selected = [item['pid'] for item in pathology_info if item['pid'] in valid]
            test_selected = [item['pid'] for item in pathology_info if item['pid'] in test]
            task_mode_counter[task] = []
            for mode, selected in zip(['train', 'valid', 'test'], [train_selected, valid_selected, test_selected]):
                labels = []
                for item in pathology_info:
                    if item['pid'] in selected:
                        labels.append(item[task])
                task_mode_counter[task].append(Counter(labels))
                
            # promise the number of each class in validset, testset is similar
            for cls, count in task_mode_counter[task][1].items():
                if abs(count - task_mode_counter[task][2][cls]) >= 5:  # 保证validset和testset中的每个类别的样本相差不超过5
                    flag = False
                    break
                
            # promise the ratio of each class in trainset, validset and testset is similar
            totals = [train_selected, valid_selected, test_selected]
            for i in range(1, 3):
                for cls, count in task_mode_counter[task][i].items():
                    a_ratio = count / len(totals[i])
                    b_ratio = task_mode_counter[task][0][cls] / len(totals[0])
                    if abs(a_ratio - b_ratio) >= 0.1:
                        flag = False
                        break
            
            if not flag:
                break
        if flag:
            break

    # add the pids with -1 label to the trainset
    for item in pathology_info:
        if item['pid'] not in legal_pids:
            train.append(item['pid'])

    # visualize the distribution of each class in each mode
    for task in tasks:
        for mode, i in zip(['train', 'valid', 'test'], range(3)):
            print(f"{task}, {mode}: {task_mode_counter[task][i]}")

    print("Times: ", start)
    print("Length of train, valid, test: ", len(train), len(valid), len(test))
    with open(f"./Data/split_seed={seed}.json", 'w') as f:
        json.dump({
            'train': train,
            'valid': valid,
            'test': test
        }, f)

