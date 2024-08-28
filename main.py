from model import CLOA
from data_module import CustomDataModule, CustomEvaluationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier
import os

def train(epochs, batch_size, dataset, pretrain_dir = None, OAR=True, 
          supervised=False, devices=1, num_workers=9, scale_start=0.08):
    if pretrain_dir != None:
        model = CLOA.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOA(batch_size, dataset, OAR, supervised, devices)
    
    data_module = CustomDataModule(batch_size=batch_size, dataset=dataset, 
                                   num_workers=num_workers, scale_start=scale_start)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Train", name=f'{dataset}-{batch_size*devices}-oar:{OAR}-scale:{scale_start}')

    #next use iNaturalist
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
    trainer.save_checkpoint(f'{dataset}-{batch_size*devices}-oar:{OAR}-scale:{scale_start}.ckpt')


def eval(pretrain_dir, batch_size, epochs, dataset, OAR, num_workers, scale_start):
    model = CLOA.load_from_checkpoint(pretrain_dir)
    data_module = CustomEvaluationDataModule(batch_size=batch_size, dataset=dataset, 
                                             num_workers=num_workers, OAR=OAR, scale_start=scale_start)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Eval", name=f'{dataset}-oar:{OAR}-scale:{scale_start}')
    if dataset == "cifar10": 
        num_classes = 10
        feature_dim = 128
    elif dataset == "cifar100": 
        num_classes = 100
        feature_dim = 128
    elif dataset == "imagenet":
        num_classes = 1000
        feature_dim = 1024

    linear_classifier = LinearClassifier(
        model, batch_size, feature_dim=feature_dim, num_classes=num_classes, topk=(1,5), freeze_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='linear_eval-{epoch:04d}',
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
    trainer.save_checkpoint(f'linear_eval-{dataset}-oar:{OAR}-scale:{scale_start}.ckpt')


def extract_data(dataset):
    data_module = CustomDataModule(batch_size=32, dataset=dataset)
    data_module.setup(stage="fit")


if __name__ == '__main__':
    seed_everything(1234) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=9)
    parser.add_argument("--scale_start", type=float, default=0.08)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--OAR", action='store_true')
    parser.add_argument("--supervised", action='store_true')
    parser.add_argument("--extract_data", action='store_true')
    args = parser.parse_args()

    if args.eval:
        eval(args.pretrain_dir, args.batch_size, args.epochs, args.dataset, args.OAR, args.num_workers, args.scale_start)
    elif args.extract_data:
        extract_data(args.dataset)
    else:
        train(args.epochs, args.batch_size, args.dataset, args.pretrain_dir, 
              args.OAR, args.supervised, args.devices, args.num_workers, args.scale_start)