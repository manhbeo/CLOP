from model import CLOA
from data_module import CustomDataModule, CustomEvaluationDataModule
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier

def train(epochs, batch_size, dataset, pretrain_dir = None, OAR=True, OAR_only=False, supervised=False, devices=1):
    if pretrain_dir != None:
        model = CLOA.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOA(batch_size, dataset, OAR, OAR_only, supervised, devices)
    
    data_module = CustomDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Train", name=f'{dataset}-{batch_size*devices}-oar:{OAR}')

    #next use iNaturalist
    trainer = pl.Trainer(logger=wandb_logger,
                        max_epochs=epochs,
                        devices="auto",
                        accelerator="gpu",
                        strategy="ddp",
                        sync_batchnorm=True,
                        use_distributed_sampler=True,
                        deterministic=True)

    trainer.fit(model, data_module)
    trainer.save_checkpoint(f'{dataset}-{batch_size*devices}-oar:{OAR}-only:{OAR_only}.ckpt')


def eval(pretrain_dir, batch_size, epochs, dataset, OAR, OAR_only):
    model = CLOA.load_from_checkpoint(pretrain_dir)
    data_module = CustomEvaluationDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Eval", name=f'linear_eval-{dataset}-oar:{OAR}')
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
        model, batch_size, feature_dim=feature_dim, num_classes=num_classes, freeze_model=True,
    )

    trainer = pl.Trainer(logger=wandb_logger,
                        max_epochs=epochs,
                        devices="auto",
                        accelerator="gpu",
                        strategy="ddp_find_unused_parameters_true",
                        sync_batchnorm=True,
                        use_distributed_sampler=True,
                        deterministic=True)
    trainer.fit(linear_classifier, datamodule=data_module)
    trainer.save_checkpoint(f'linear_eval-{dataset}-oar:{OAR}-only:{OAR_only}.ckpt')


def extract_data(dataset):
    data_module = CustomDataModule(batch_size=32, dataset=dataset)
    data_module.setup(stage="fit")

def test(pretrain_dir, dataset):
    model = CLOA.load_from_checkpoint(pretrain_dir)
    data_module = CustomEvaluationDataModule(batch_size=128, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_test")

    trainer = pl.Trainer(logger=wandb_logger,
                        devices="auto",
                        accelerator="gpu",
                        sync_batchnorm=True,
                        use_distributed_sampler=True,
                        deterministic=True)
    trainer.test(model, data_module)


if __name__ == '__main__':
    seed_everything(1234) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain_dir", type=str)
    parser.add_argument("--pretrain_linear_classifier_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--OAR", action='store_true')
    parser.add_argument("--OAR_only", action='store_true')
    parser.add_argument("--supervised", action='store_true')
    parser.add_argument("--extract_data", action='store_true')
    args = parser.parse_args()

    if args.eval:
        eval(args.pretrain_dir, args.batch_size, args.epochs, args.dataset, args.devices, args.OAR, args.OAR_only, args.supervised)
    elif args.extract_data:
        extract_data(args.dataset)
    elif args.test:
        test(args.pretrain_dir, args.dataset)
    else:
        train(args.epochs, args.batch_size, args.dataset, args.pretrain_dir, args.OAR, args.OAR_only, args.supervised, args.devices)