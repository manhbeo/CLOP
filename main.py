from model import CLOP
from data_module import CustomDataModule, CustomEvaluationDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
from pytorch_lightning import seed_everything
from linear_classifier import LinearClassifier
import torch.nn as nn
from object_detection import ObjectDetectionClassifier

def read_class_percentages(file_path: str):
    """Read class percentages from file and parse them
    Expected file format: '0:0.5,1:1.0,2:0.75'
    """
    if not file_path:
        return None
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        # Parse directly without curly braces
        return {int(k): float(v) for k, v in (i.split(':') for i in content.split(','))}
    except FileNotFoundError:
        print(f"Warning: Class percentage file '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Warning: Error reading class percentages: {e}")
        return None

def train(epochs, batch_size, dataset, pretrain_dir = None, has_CLOP=True, loss="nxt_ent", devices=1, k=100, num_workers=9, 
          distance="cosine",lr=None, lambda_val=1.0, class_per_dir= None, label_per=1.0, etf=False, semi=False):
    if pretrain_dir != None:
        model = CLOP.load_from_checkpoint(pretrain_dir)
    else: 
        model = CLOP(batch_size, dataset, has_CLOP, loss, devices, k, distance, lr, lambda_val, etf, semi)

    class_per = read_class_percentages(class_per_dir)
    data_module = CustomDataModule(batch_size=batch_size, dataset=dataset, num_workers=num_workers, loss=loss,
                                   class_percentages= class_per, label_percentage=label_per)
    wandb_logger = pl.loggers.WandbLogger(project="CLOA_Train", name=f'{dataset}-{batch_size*devices}-{loss}-CLOP={has_CLOP}')

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
    trainer.save_checkpoint(f'{batch_size*devices}-{loss}.ckpt')


def eval(pretrain_dir: str,
         batch_size: int,
         epochs: int,
         dataset_type: str,
         num_workers: int = 9,
         lr: float = None,
         tasks: list = None):
    """
    Evaluate pretrained model on specified downstream tasks

    Args:
        pretrain_dir: Path to pretrained model checkpoint
        batch_size: Batch size for training
        epochs: Number of training epochs
        dataset_type: Dataset type/name to evaluate on
        num_workers: Number of data loading workers
        lr: Learning rate (optional)
        tasks: List of tasks to evaluate on ['linear', 'detection'] (default: ['linear'])
    """
    if tasks is None:
        tasks = ['linear']

    results = {}
    model = CLOP.load_from_checkpoint(pretrain_dir)

    # Get number of classes based on dataset
    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'food101': 101,
        'birdsnap': 500,
        'sun397': 397,
        'stanfordcars': 196,
        'aircraft': 100,
        'dtd': 47,
        'voc2007': 20,
        'oxfordpets': 37,
        'caltech101': 101,
        'flowers102': 102
    }.get(dataset_type)

    if num_classes is None:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    # Linear Classification Evaluation
    if 'linear' in tasks:
        model_linear = model
        model_linear.projection_head = nn.Identity()

        data_module = CustomEvaluationDataModule(
            batch_size=batch_size,
            dataset_type=dataset_type,
            task='classification',
            num_workers=num_workers
        )

        wandb_logger = pl.loggers.WandbLogger(
            project="CLOA_Eval",
            name=f'linear-{dataset_type}-{pretrain_dir.split("/")[-1].replace(".ckpt", "")}'
        )

        data_module.setup(stage='fit')

        linear_classifier = LinearClassifier(
            model_linear,
            batch_size,
            num_classes=num_classes,
            freeze_model=True,
            lr=lr
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode="min",
            dirpath='checkpoints/linear',
            filename=f'linear_{dataset_type}-{{epoch:03d}}-{{val_loss:.4f}}',
            save_weights_only=True,
            every_n_epochs=1,
            verbose=True
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=epochs,
            devices="auto",
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            sync_batchnorm=True,
            use_distributed_sampler=True,
            callbacks=[checkpoint_callback],
            deterministic=True
        )

        trainer.fit(linear_classifier, datamodule=data_module)
        final_path = f'checkpoints/linear/final_{dataset_type}_{pretrain_dir.split("/")[-1]}'
        trainer.save_checkpoint(final_path)
        results['linear'] = final_path

    # Detection Evaluation
    if 'detection' in tasks and dataset_type in ['voc2007', 'oxfordpets', 'caltech101', 'flowers102']:
        model_detection = model

        data_module = CustomEvaluationDataModule(
            batch_size=batch_size,
            dataset_type=dataset_type,
            task='detection',
            num_workers=num_workers
        )

        wandb_logger = pl.loggers.WandbLogger(
            project="CLOA_Eval_Detection",
            name=f'detection-{dataset_type}-{pretrain_dir.split("/")[-1].replace(".ckpt", "")}'
        )

        data_module.setup(stage='fit')

        detector = ObjectDetectionClassifier(
            model_detection,
            batch_size,
            num_classes=num_classes,
            freeze_backbone=True,
            lr=lr
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode="min",
            dirpath='checkpoints/detection',
            filename=f'detection_{dataset_type}-{{epoch:03d}}-{{val_loss:.4f}}',
            save_weights_only=True,
            every_n_epochs=1,
            verbose=True
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=epochs,
            devices="auto",
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            sync_batchnorm=True,
            use_distributed_sampler=True,
            callbacks=[checkpoint_callback],
            deterministic=True
        )

        trainer.fit(detector, datamodule=data_module)
        final_path = f'checkpoints/detection/final_{dataset_type}_{pretrain_dir.split("/")[-1]}'
        trainer.save_checkpoint(final_path)
        results['detection'] = final_path

    return results


def extract_data(dataset):
    data_module = CustomDataModule(batch_size=32, dataset=dataset)
    data_module.setup(stage="fit")


if __name__ == '__main__':
    seed_everything(1234) 
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
    parser.add_argument("--class_per_file",
                        type=str,
                        default=None,
                        help='Path to file containing class percentages in format "class_idx:percentage,...". Example file content: "0:0.5,1:1.0"')
    parser.add_argument("--loss", type=str)
    parser.add_argument("--has_CLOP", action='store_true')
    parser.add_argument("--extract_data", action='store_true')
    parser.add_argument("--etf", action='store_true')
    parser.add_argument("--semi", action='store_true')
    parser.add_argument("--task", type=str, default="linear")
    args = parser.parse_args()

    if args.eval:
        eval(args.pretrain_dir, args.batch_size, args.epochs, args.dataset, args.num_workers, args.lr, args.task)
    elif args.extract_data:
        extract_data(args.dataset)
    else:
        train(args.epochs, args.batch_size, args.dataset, args.pretrain_dir, args.has_CLOP, args.loss, args.devices, args.k, 
              args.num_workers, args.distance, args.lr, args.lambda_val, args.class_per_file, args.label_por, args.etf, args.semi)