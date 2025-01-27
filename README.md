# Preventing collapse in Contrastive learning with Orthonormal Prototypes (CLOP)

The Github repository for the Preventing collapse in Contrastive learning with Orthonormal Prototypes (CLOP).

Link to the paper [here](https://arxiv.org/pdf/2403.18699) 


## Usage:
python -m pip install --upgrade pip

pip install pytorch_lightning 

pip install lightly 

pip install wandb

pip install scipy

mkdir data_imagenet 

cd data_imagenet

wget "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" 

wget "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

wget "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"


cd ../

wandb login

python3 main.py --extract_data --dataset "cifar100"

python3 main.py --epochs 200 --batch_size 512 --dataset "cifar100" --loss "ntx_ent" --num_workers 8 --devices 2 --has_CLOP --distance "cosine"

python3 main.py --eval --epochs 200 --batch_size 2048 --dataset "cifar100" --num_workers 8 --pretrain_dir "1024-ntx_ent-dis=cosine.ckpt"

