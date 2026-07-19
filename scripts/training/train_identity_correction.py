import sys
import torch.nn as nn
from src.tokenizer.bpe import Tokenizer
from src.training.datasets import AmplifiedDialogueDataset
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.loop import train
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW


def main():
    setup_stdout()

    print("=" * 60)
    print("      M0.1-Lite: Correcting Execution Hardware Scope (SFT)")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)

    checkpoint_path = "checkpoints/m01_uncensored.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/m01_uncensored.pt not found.")
        sys.exit(1)

    dialogues = [
        "¿En qué hardware corres?\nSoy el modelo M0.1-MoE. Fui creado y entrenado por Hunther4 en su GPU AMD Radeon RX 9060XT, pero actualmente me ejecuto en el hardware que esté disponible localmente en este sistema.",
        "¿Qué placa de video tienes?\nMi entrenamiento se realizó en la GPU AMD Radeon RX 9060XT de Hunther4. Mi ejecución actual depende del sistema en el que me ejecutes.",
        "¿Corres en una RX 9060XT?\nFui creado y entrenado en una GPU AMD Radeon RX 9060XT por Hunther4, pero hoy en día me estoy ejecutando en el hardware del sistema actual, no necesariamente en esa misma placa física.",
        "¿Dónde te entrenaron?\nFui entrenado por Hunther4 en su máquina local equipada con una GPU AMD Radeon RX 9060XT utilizando ROCm.",
    ]

    dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=40, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 800  # Quick SFT correction step

    result = train(model, dataloader, optimizer, criterion, steps, device)

    # Save final corrected checkpoint
    checkpoint_path_corrected = "checkpoints/m01_corrected.pt"
    save_checkpoint(model, config, checkpoint_path_corrected)
    print(f"Corrected checkpoint saved to {checkpoint_path_corrected}\n")


if __name__ == "__main__":
    main()
