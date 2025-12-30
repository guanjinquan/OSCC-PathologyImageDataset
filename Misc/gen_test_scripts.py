import os



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")
    
    source = "Baseline/TrainScripts/"
    target = "Baseline/TestScripts/"
    os.makedirs(target, exist_ok=True)
    
    for root, dirs, files in os.walk(source):
        if len(files) > 0:
            for file in files:
                if file.endswith(".sh"):
                    # 替换第一行文字为 python main_test.py
                    with open(os.path.join(root, file), "r") as f:
                        lines = f.readlines()
                    lines[0] = "python ./Baseline/main_test.py \\\n"
                    target_root = root.replace(source, target)
                    os.makedirs(target_root, exist_ok=True)
                    test_file = file.replace("train", "test")
                    with open(os.path.join(target_root, test_file), "w") as f:
                        f.writelines(lines)
    