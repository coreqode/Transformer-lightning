from config import replace, preEnc, preEncDec
# from train_util import run_train
import model as M
from data import IndicDataset, PadSequence
from time import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import time
import torch
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME
import torch_optimizer as optim
from pytorch_lightning.loggers import TensorBoardLogger

def gen_model_loaders(config):
    model, tokenizers = M.build_model(config)
    pad_sequence = PadSequence(
        tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)
    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True),
                              batch_size=config.batch_size,
                              shuffle=False,
                              collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False),
                             batch_size=config.eval_size,
                             shuffle=False,
                             collate_fn=pad_sequence)
    return model, tokenizers, train_loader, eval_loader

class BertModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.init_seed()
        self.config = preEncDec
        self.model, self.tokenizers, self.train_loader, self.val_loader = gen_model_loaders(self.config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        source, target = batch
        loss, logits = self.model(source, target)
        log = {'train_loss': loss}
        return {'loss': loss, 'log':log}
    
    def validation_step(self, batch, batch_idx):
        source, target = batch
        loss, logits = self.model(source, target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
        tmp_eval_accuracy = torch.tensor(tmp_eval_accuracy)
        return {'val_loss': loss, 'val_acc': tmp_eval_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return { 'log':log}

    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        #print (f'preds: {pred_flat}')
        #print (f'labels: {labels_flat}')
        return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        optimizer = optim.RAdam(self.model.parameters(),lr= self.config.lr,betas=(0.9, 0.999),eps=1e-8,weight_decay=0,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_loader), eta_min=self.config.lr)
        return [optimizer], [scheduler]

    def init_seed(self):
        seed_val = 42
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def save_model(self, model, output_dir):
        output_dir = Path(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = output_dir / WEIGHTS_NAME
        output_config_file = output_dir / CONFIG_NAME
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #src_tokenizer.save_vocabulary(output_dir)


if __name__ == '__main__':
    log_path = '/content/content/itr/logs/pretrained-enc-dec'
    model = BertModel()
    logger = TensorBoardLogger(save_dir=log_path,version=1,name='lightning_logs')
    trainer = pl.Trainer(max_epochs=2,gpus=1, logger=logger, show_progress_bar=True)
    trainer.fit(model)