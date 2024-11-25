# Enviroment

```shell
pip install -r enviroment.txt
pip install mambavision
pip install mamba-ssm[causal-conv1d]
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install .
cd ..
```


#  TRAIN



```shell
nohup python Train.py --train_root /workspace/CODlab/data/datasets/COD10K_train/train/ \
 --val_root /workspace/CODlab/data/datasets/COD10K_train/test/ \
 --save_path /workspace/CODlab/Pths/result/ADD_PFAFM/ > Baseline.log 2>&1 &
```

 # TEST



 ```shell
 python Test.py --pth_path /workspace/CODlab/FSEL/FSEL_ECCV_2024/save_dataNet_epoch_best.pth \
 --test_dataset_path /workspace/CODlab/data/datasets/CAMO
```