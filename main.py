from model import CLOA
from data_module import CIFARDataModule, CIFAREvaluationDataModule#, ImageNetDataModule, ImageNetEvaluationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier
import torch

#TODO: start with adamw
def sweep(args):
    def sweep_train():
      wandb.init()
      data_module = CIFARDataModule(batch_size=wandb.config.batch_size, dataset=args.dataset)
      model = CLOA(
          learning_rate=wandb.config.learning_rate,
          lr_schedule=wandb.config.lr_schedule,
          optimizer=wandb.config.optimizer,
          temperature=wandb.config.temperature,
          lambda_val=wandb.config.lambda_val,
          feature_bank_size=wandb.config.batch_size
      )

      wandb_logger = pl.loggers.WandbLogger(project="CLOA_Sweep")

      checkpoint_callback = ModelCheckpoint(
          monitor='val_loss',
          dirpath="",
          filename='loss:{val_loss:.5f}',
          save_top_k=1,
          mode='min',
          every_n_epochs=10)

      trainer = pl.Trainer(logger=wandb_logger,
                    devices="auto",
                    max_epochs = args.epochs,
                    accelerator="auto",
                    strategy="ddp",
                    sync_batchnorm=True,
                    use_distributed_sampler=True,
                    callbacks=[checkpoint_callback],
                    deterministic=True)

      trainer.fit(model, data_module)
    return sweep_train



def train(learning_rate, optimizer, lr_schedule, temperature, lambda_val, epochs, batch_size, dataset, pretrain_dir = None, have_coarse_label=True):
    if pretrain_dir != None: #if pretrain_dir exist
        model = CLOA.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOA(learning_rate, lr_schedule, optimizer, temperature, lambda_val, batch_size, dataset, have_coarse_label)
    
    # if dataset == "imagenet":
    #     data_module = ImageNetDataModule(batch_size=batch_size)
    #     wandb_logger = pl.loggers.WandbLogger(project="Train_Tree_CLR_ImageNet")
    # elif dataset == "cifar100": #TODO: fix to begin with
    data_module = CIFARDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Train")

    #Save the model after each 5 epochs
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='model-{epoch:04d}-{val_acc:.2f}',
        save_weights_only=True,
        every_n_epochs=3,
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

    # if dataset == "cifar100":
    data_module = CIFAREvaluationDataModule(batch_size=batch_size, dataset=dataset)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Eval")
    if dataset == "cifar100": num_classes = 100
    elif dataset == "cifar10": num_classes = 10
    # elif dataset == "imagenet":
    #     data_module = ImageNetEvaluationDataModule(batch_size=batch_size)
    #     wandb_logger = pl.loggers.WandbLogger(project="Eval_Tree_CLR_ImageNet")
    #     num_classes = 1000

    
    linear_classifier = LinearClassifier(
        model, batch_size, feature_dim=128, num_classes=num_classes, topk=(1,5), freeze_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath='',
        filename='model-{epoch:04d}-{val_top1:.2f}',
        save_weights_only=True,
        every_n_epochs=3,
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

    # Train the classifier
    trainer.fit(linear_classifier, datamodule=data_module)

def test(pretrain_dir, pretrain_linear_classifier_dir, batch_size, dataset):
    # if dataset == "cifar100":
    data_module = CIFAREvaluationDataModule(batch_size=batch_size)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Test")
    if dataset == "cifar100": num_classes = 100
    elif dataset == "cifar10": num_classes = 10
    # elif dataset == "imagenet":
    #     data_module = CIFAR100EvaluationDataModule(batch_size=batch_size)
    #     wandb_logger = pl.loggers.WandbLogger(project="Test_Tree_CLR_ImgNet")
    #     num_classes = 1000

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
        model, batch_size, feature_dim=128, num_classes=num_classes, topk=(1,5), freeze_model=True
    )
    linear_classifier.load_state_dict(torch.load(pretrain_linear_classifier_dir)['state_dict'])

    
    trainer.test(linear_classifier, datamodule=data_module)


if __name__ == '__main__':
    seed_everything(1234) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--lr", type=float, default=1.2)
    parser.add_argument("--opt", type=str, default="lars")
    parser.add_argument("--lr_schedule", type=str, default="exp")
    parser.add_argument("--temp", type=float, default=0.192)
    parser.add_argument("--have_coarse_label", action='store_true')
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain_model", type=str)
    parser.add_argument("--pretrain_linear_classifier_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="imagenet")


    args = parser.parse_args()

    if args.sweep:
        sweep_train_config = {
            'method': 'random',
            'metric': {
                'name': 'val_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'min': 0.1,
                    'max': 6.5
                },
                'temperature': {
                    "min":0.1,
                    "max":1.0
                },
                'lambda_val': {
                    "min":0.0,
                    "max":1.0
                },
                'optimizer':{
                    'values': ['sgd', 'adam', 'adamw']
                },
                'lr_schedule':{
                    'values':['linear', 'cosine', 'exp']
                },
                'batch_size': {
                    'values': [256, 512, 1024]
                }
            }
        }
        sweep_id = wandb.sweep(sweep_train_config, project="Sweep_Tree_CLR")
        wandb.agent(sweep_id, sweep(args))

    elif args.eval:
        eval(args.pretrain_model, args.batch_size, args.epochs, args.dataset)
    elif args.test:
        test(args.pretrain_model, args.pretrain_linear_classifier_dir, args.batch_size, args.dataset)
    else:
        train(args.lr, args.opt, args.lr_schedule, args.temp, args.lambda_val, args.epochs, args.batch_size, args.dataset, args.pretrain_model, args.have_coarse_label)