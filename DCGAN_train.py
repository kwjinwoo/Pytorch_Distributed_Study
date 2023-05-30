import torch
from models import Generator, Discriminator
from utils import load_config, seed_everything, save_generated_image
from trainutils import SingleTrainer
from datasets import load_dataloader


config_path = "./configs/config.yaml"
config = load_config(config_path)

seed_everything(config["seed"])

train_data = load_dataloader(config["data_root"], config["image_size"],
                             config["batch_size"], config["workers"])

device = 0   # shorthand for cuda:0
generator = Generator(config["nz"], config["ngf"])
discriminator = Discriminator(config["ndf"])

optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr"])
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

criterion = torch.nn.BCELoss()

trainer = SingleTrainer(generator, discriminator, train_data, optimizer_g, optimizer_d,
                        criterion, device, use_wandb=True, config=config)

trainer.train(config["num_epochs"])
save_generated_image(trainer.generator, config["nz"], device)
