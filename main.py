from model import CLOP
from data_module import CustomDataModule, CustomEvaluationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier
import torch.nn as nn

def train(epochs, batch_size, dataset, pretrain_dir = None, CLOP=True, loss="nxt_ent", devices=1, k=100, num_workers=9, 
          distance="cosine", augment="auto_imgnet", lr=None, lambda_val=1.0, label_por=1.0):
    if pretrain_dir != None:
        model = CLOP.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOP(batch_size, dataset, CLOP, loss, devices, k, distance, lr, lambda_val, label_por) 
    
    data_module = CustomDataModule(batch_size=batch_size, dataset=dataset, num_workers=num_workers, augment=augment, loss=loss)
    wandb_logger = pl.loggers.WandbLogger(project="CLOP_Train", name=f'{dataset}-{batch_size*devices}-CLOP={CLOP}-label={label_por}')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='{epoch:03d}',
        save_weights_only=True,
        every_n_epochs=1,
        verbose=True
    )

    trainer = pl.Trainer(logger=wandb_logger,
                        max_epochs=epochs,
                        devices="auto",
                        accelerator="gpu",
                        strategy="ddp",
                        sync_batchnorm=True,
                        use_distributed_sampler=True,
                        callbacks=[checkpoint_callback],
                        deterministic=True)

    trainer.fit(model, data_module)
    trainer.save_checkpoint(f'{batch_size*devices}-CLOP={CLOP}-label={label_por}.ckpt')


def eval(pretrain_dir, batch_size, epochs, dataset, num_workers, augment="auto_imgnet", label_por=1.0, lr=None):
    model = CLOP.load_from_checkpoint(pretrain_dir)
    model.projection_head = nn.Identity()
    data_module = CustomEvaluationDataModule(batch_size=batch_size, dataset=dataset, num_workers=num_workers, augment=augment, subset_fraction=label_por)

    wandb_logger = pl.loggers.WandbLogger(project="CLOP_Eval", name=f'{dataset}-{pretrain_dir[:-5]}')
    if dataset == "cifar10": 
        num_classes = 10
    elif dataset == "cifar100": 
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000
    elif dataset == "tiny_imagenet":
        num_classes = 200

    data_module.setup(stage='fit')

    linear_classifier = LinearClassifier(
        model, batch_size, num_classes=num_classes, freeze_model=True, lr=lr
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='linear_eval-{epoch:03d}',
        save_weights_only=True,
        every_n_epochs=1,
        verbose=True
    )

    trainer = pl.Trainer(logger=wandb_logger,
                        max_epochs=epochs,
                        devices="auto",
                        accelerator="gpu",
                        strategy="ddp_find_unused_parameters_true",
                        sync_batchnorm=True,
                        use_distributed_sampler=True,
                        callbacks=[checkpoint_callback],
                        deterministic=True)
    trainer.fit(linear_classifier, datamodule=data_module)
    trainer.save_checkpoint(f'linear_eval-{pretrain_dir}')


def extract_data(dataset):
    data_module = CustomDataModule(batch_size=32, dataset=dataset)
    data_module.setup(stage="fit")


if __name__ == '__main__':
    seed_everything(123) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda_val", type=float, default=1.0)
    parser.add_argument("--label_por", type=float, default=1.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("-k", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=9)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--augment", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--CLOP", action='store_true')
    parser.add_argument("--extract_data", action='store_true')
    args = parser.parse_args()

    if args.eval:
        eval(args.pretrain_dir, args.batch_size, args.epochs, args.dataset, args.num_workers, args.augment, args.label_por, args.lr)
    elif args.extract_data:
        extract_data(args.dataset)
    else:
        train(args.epochs, args.batch_size, args.dataset, args.pretrain_dir, args.CLOP, args.loss, args.devices, args.k, 
              args.num_workers, args.distance, args.augment, args.lr, args.lambda_val, args.label_por)