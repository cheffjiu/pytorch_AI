import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
        logger,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
            self.logger.update(outputs, labels)

        avg_loss = total_loss / len(self.train_loader)
        self.logger.compute_and_log(prefix="train")
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                self.logger.update(outputs, labels)

        avg_loss = total_loss / len(self.val_loader)
        self.logger.compute_and_log(prefix="val")
        return avg_loss
