import os


"""
dependencies:
 - python >= 3.10
 - huggeringface-cli
"""

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__) + "/../")
    os.makedirs("pretrained_weight", exist_ok=True)
    
    # git lfs install
    # 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
    os.system("git lfs install")
    
    # download hibou-b from https://huggingface.co
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/models--histai--hibou-b"):
        os.system("python ./ExternalLibs/HuggingFace-Download-Accelerator/hf_download.py --model histai/hibou-b --save_dir ./Baseline/models/backbones/pretrained_weight/")
        
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/models--paige-ai--Virchow2"):
        os.system("python ./ExternalLibs/HuggingFace-Download-Accelerator/hf_download.py --model paige-ai/Virchow2 --save_dir ./Baseline/models/backbones/pretrained_weight/")
    
    # download medcoss 
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/medcoss-epch299.pth"):
        print("Visiting https://github.com/yeerwen/MedCoSS, download mannually.", flush=True)
        
    # down git clone UNI
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/models--mahmoodlab--UNI"):
        # os.system("git clone https://huggingface.co/MahmoodLab/UNI")
        os.system("python ./ExternalLibs/HuggingFace-Download-Accelerator/hf_download.py --model MahmoodLab/UNI --save_dir ./Baseline/models/backbones/pretrained_weight/")
    
    # download git clone https://huggingface.co/prov-gigapath/prov-gigapath
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/models--prov-gigapath--prov-gigapath"):
        # os.system("git clone https://huggingface.co/prov-gigapath/prov-gigapath")
        os.system("python ./ExternalLibs/HuggingFace-Download-Accelerator/hf_download.py --model prov-gigapath/prov-gigapath --save_dir ./Baseline/models/backbones/pretrained_weight/")
    
    # download git clone https://huggingface.co/MahmoodLab/CONCH
    if not os.path.exists("./Baseline/models/backbones/pretrained_weight/models--mahmoodlab--CONCH"):
        # os.system("git clone https://huggingface.co/MahmoodLab/CONCH")
        os.system("python ./ExternalLibs/HuggingFace-Download-Accelerator/hf_download.py --model MahmoodLab/CONCH --save_dir ./Baseline/models/backbones/pretrained_weight/")
    