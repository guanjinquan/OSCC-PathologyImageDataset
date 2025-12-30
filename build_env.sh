# please run these commands manually in your terminal
conda env create -f environment.yml
conda activate Multi-OSCC
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
