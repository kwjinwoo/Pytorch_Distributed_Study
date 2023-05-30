import torch
import wandb
from datetime import datetime


class SingleTrainer:
    def __init__(self, generator, discriminator, train_data, optimizer_g, optimizer_d, criterion, gpu_id,
                 use_wandb=False, config=None):
        self.gpu_id = gpu_id
        self.generator = torch.compile(generator.to(gpu_id))
        self.discriminator = torch.compile(discriminator.to(gpu_id))
        self.train_data = train_data
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion = criterion

        if use_wandb:
            wandb.init(project="Pytorch Distributed Study",
                       config=config)
            wandb.run.name = "SingleGPU_" + str(datetime.now())

    # def _run_batch(self):
    def _run_epoch(self):
        G_losses = 0
        D_losses = 0
        for data in self.train_data:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            self.discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.gpu_id)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=self.gpu_id)
            # Forward pass real batch through D
            output = self.discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.generator.nz, 1, 1, device=self.gpu_id)
            # Generate fake image batch with G
            fake = self.generator(noise)
            label.fill_(0.)
            # Classify all fake batch with D
            output = self.discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizer_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.generator.zero_grad()
            label.fill_(1.)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            self.optimizer_g.step()

            G_losses += errG.item()
            D_losses += errD.item()
        return G_losses / len(self.train_data), D_losses / len(self.train_data)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            g_loss, d_loss = self._run_epoch()
            wandb.log({
                "generator_loss": g_loss,
                "discriminator_loss": d_loss
            })
