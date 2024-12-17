import os
import json
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import re
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn

# Regular expression matching whitespace
_whitespace_re = re.compile(r'\s+')

def collapse_whitespace(text):
    """여러 공백을 하나로 축소"""
    return re.sub(_whitespace_re, ' ', text).strip()

def korean_cleaners(text):
    """
    한국어 전처리 함수.
    - 특수문자 제거
    - 공백 정리
    """
    text = re.sub(r"[^가-힣0-9\s.,!?]", "", text)  # 한글, 숫자, 기본 특수문자만 허용
    text = collapse_whitespace(text)  # 공백 정리
    return text

def basic_cleaners(text):
    """소문자 변환 및 공백 축소"""
    text = text.lower()  # 한국어에는 필요 없으나 영어 섞인 데이터가 있을 경우 사용
    text = collapse_whitespace(text)
    return text

# Define symbols
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가나다라마바사아자차카타파하'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Enable cuDNN auto-tuning for performance
torch.backends.cudnn.benchmark = True

# Global step tracker for training
global_step = 0

# Set working directory
os.chdir("C:/Users/user/Desktop/furina_project/vits")
print("Current Working Directory:", os.getcwd())

def main():
    """Main training loop."""
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: GPU not available, using CPU instead.")

    # Load hyperparameters
    hps = utils.get_hparams()

    print(f"Using device: {device}")
    run_training(0, 1, hps, device)

def run_training(rank, n_gpus, hps, device):
    """Training logic."""
    global global_step

    # Set seed for reproducibility
    torch.manual_seed(hps.train.seed)
    if device == "cuda":
        torch.cuda.set_device(rank)

    # Load dataset
    print("Loading dataset...")
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    print("Initializing model...")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_heads=hps.model.n_heads if hasattr(hps.model, 'n_heads') else 2,
        n_layers=hps.model.n_layers if hasattr(hps.model, 'n_layers') else 6,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes if hasattr(hps.model, 'resblock_dilation_sizes') else [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        **{k: v for k, v in vars(hps.model).items() if k not in ["n_heads", "n_layers", "resblock_dilation_sizes"]}
    ).to(device)

    # Initialize optimizer
    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # GradScaler for mixed precision training
    scaler = GradScaler(enabled=hps.train.fp16_run)

    # Training loop
    for epoch in range(1, hps.train.epochs + 1):
        print(f"Starting epoch {epoch}...")
        net_g.train()

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            # Move data to device
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

            # Forward and backward pass with autocast for mixed precision
            with autocast(enabled=hps.train.fp16_run):
                y_hat, *_ = net_g(x, x_lengths, spec, spec_lengths)
                loss = F.l1_loss(y, y_hat)

            # Backpropagation and optimization
            optim_g.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

        print(f"Epoch {epoch} completed. Loss: {loss.item()}")

        # Save model checkpoints
        save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, hps.train.model_dir)

def save_checkpoint(model, optimizer, learning_rate, epoch, model_dir):
    """Save model checkpoint."""
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"G_{epoch}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
        "epoch": epoch
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
