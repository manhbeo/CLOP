from model import CLOA
from data_module import CustomDataModule, CustomEvaluationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier
import torch

def train(epochs, batch_size, dataset, pretrain_dir = None, OAR=True, OAR_only=False, supervised=False):
    if pretrain_dir != None: #if pretrain_dir exist
        model = CLOA.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOA(batch_size, dataset, OAR, OAR_only, supervised)
    
    data_module = CustomDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Train")

    #Save the model after each 1 epoch
    #next use iNaturalist
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='256_oar_sup-{epoch:04d}',
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


def eval(pretrain_dir, batch_size, epochs, dataset):
    model = CLOA.load_from_checkpoint(pretrain_dir)
    data_module = CustomEvaluationDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Eval")
    if dataset == "cifar100": 
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
        filename='{model}-{epoch:04d}',
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


def test(pretrain_dir, pretrain_linear_classifier_dir, batch_size, dataset):
    data_module = CustomEvaluationDataModule(batch_size=batch_size)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Test")
    if dataset == "cifar100": 
        num_classes = 100
        feature_dim = 128
    elif dataset == "imagenet":
        num_classes = 1000
        feature_dim = 1024

    trainer = pl.Trainer(logger=wandb_logger,
                    devices="auto",
                    accelerator="gpu",
                    strategy="ddp_find_unused_parameters_true",
                    sync_batchnorm=True,
                    use_distributed_sampler=True,
                    deterministic=True)
    
    data_module.prepare_data()
    data_module.setup("test")
    
    model = CLOA.load_from_checkpoint(pretrain_dir)
    linear_classifier = LinearClassifier(
        model, batch_size, feature_dim=feature_dim, num_classes=num_classes, topk=(1,5), freeze_model=True,
    )
    linear_classifier.load_state_dict(torch.load(pretrain_linear_classifier_dir)['state_dict'])

    
    trainer.test(linear_classifier, datamodule=data_module)


if __name__ == '__main__':
    seed_everything(1234) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain_model", type=str)
    parser.add_argument("--pretrain_linear_classifier_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--OAR", action='store_true')
    parser.add_argument("--OAR_only", action='store_true')
    parser.add_argument("--supervised", action='store_true')
    args = parser.parse_args()

    if args.eval:
        eval(args.pretrain_model, args.batch_size, args.epochs, args.dataset, args.criterion)
    elif args.test:
        test(args.pretrain_model, args.pretrain_linear_classifier_dir, args.batch_size, args.dataset)
    else:
        train(args.epochs, args.batch_size, args.dataset, args.pretrain_model, args.OAR, args.OAR_only, args.supervised)