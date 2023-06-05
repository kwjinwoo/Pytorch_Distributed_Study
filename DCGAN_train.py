import torch
from models import Generator, Discriminator
from utils import load_config, seed_everything, save_generated_image, ddp_setup
from trainutils import SingleTrainer, DPTrainer, DDPTrainer
from datasets import load_dataloader
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="DCGAN Training script")
parser.add_argument('--mode', type=str, required=True, help="gpu training mode")

args = parser.parse_args()


def ddp_run(rank, world_size, total_epochs, generator, discriminator,
            optimizer_g, optimizer_d, criterion, config):
    ddp_setup(rank, world_size)
    train_data = load_dataloader(config["data_root"], config["image_size"],
                                 config["batch_size"], config["workers"], ddp=True)
    trainer = DDPTrainer(generator, discriminator, train_data, optimizer_g, optimizer_d,
                         criterion, rank, True, config)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    config_path = "./configs/config.yaml"
    config = load_config(config_path)

    seed_everything(config["seed"])

    train_data = load_dataloader(config["data_root"], config["image_size"],
                                 config["batch_size"], config["workers"])

    generator = Generator(config["nz"], config["ngf"])
    discriminator = Discriminator(config["ndf"])

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

    criterion = torch.nn.BCELoss()

    if args.mode == "single":
        train_data = load_dataloader(config["data_root"], config["image_size"],
                                     config["batch_size"], config["workers"])
        device = 0   # shorthand for cuda:0
        trainer = SingleTrainer(generator, discriminator, train_data, optimizer_g, optimizer_d,
                                criterion, device, use_wandb=True, config=config)
        trainer.train(config["num_epochs"])
        save_generated_image(trainer.generator, config["nz"], device)
    elif args.mode == "DDP":
        generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        mp.spawn(ddp_run, args=(config["ngpu"], config["num_epochs"], generator, discriminator,
                                optimizer_g, optimizer_d, criterion, config),
                 nprocs=config["ngpu"])

        generator.load_state_dict(torch.load("./saved/checkpoint.pt"))
        generator = generator.to(0)
        save_generated_image(generator, generator.nz, 0)

