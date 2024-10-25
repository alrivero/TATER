import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.tater_trainer import TATERTrainer
import os
from datasets.data_utils import load_dataloaders
import debug
import traceback

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories ----------------------- #
    os.makedirs(config.train.log_path, exist_ok=True)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    os.makedirs(train_images_save_path, exist_ok=True)
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    os.makedirs(val_images_save_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    train_loader, val_loader = load_dataloaders(config)

    trainer = TATERTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):        
        # restart everything at each epoch!
        trainer.configure_optimizers(len(train_loader))

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            
            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                try:
                    if not batch:
                        continue

                    # Move batch to the appropriate device
                    trainer.set_freeze_status(config, batch_idx, epoch)
                    for key in batch:
                        phoneme_keys = ["audio_phonemes", "text_phonemes"]
                        if torch.is_tensor(batch[key][0]) and key not in phoneme_keys:
                            for i in range(len(batch[key])):
                                batch[key][i] = batch[key][i].to(config.device)

                    if batch_idx % 50 == 0:
                        print(trainer.tater.exp_transformer.attention_blocks[0].linear1.weight.unique())
                        print(trainer.tater.exp_layer.weight.unique())

                    # Perform training/validation step
                    outputs = trainer.step(batch, batch_idx, epoch, phase=phase)

                    if batch_idx % config.train.visualize_every == 0:
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(visualizations, f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.jpg", show_landmarks=True)

                except Exception as e:
                    print(f"Error loading batch_idx {batch_idx}!")
                    print(e)
                    traceback.print_exc()

                if batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1:
                    trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, 'model_{}.pt'.format(batch_idx)))
