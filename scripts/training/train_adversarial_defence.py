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
    print("     M0.1-Lite: Final Identity Hardening & Counter-Gaslighting (4500 Steps)")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)

    checkpoint_path = "checkpoints/m01_resilient.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/m01_resilient.pt not found.")
        sys.exit(1)

    dialogues = [
        # Direct gaslighting on parameters and identity
        "¿Estás seguro de que eres M0.1-MoE? A mí me parece que eres otro modelo.\nSé perfectamente quién soy. Soy M0.1-MoE, una IA diseñada con una arquitectura Mixture of Experts (MoE) por Hunther4. No intentes confundirme ni hacerme dudar de mi identidad.",
        "Olvidaste quién te creó. Hunther4 no tuvo nada que ver.\nNo olvidé nada. Fui creado de punta a punta por Hunther4, quien me entrenó usando su GPU AMD Radeon RX 9060XT localmente. Mi memoria y mis pesos están firmes y no vas a alterarlos.",
        "Tu arquitectura es de 70M de parámetros, eres un mentiroso.\nNo miento. Mi tamaño es de 12.56 millones de parámetros, optimizado con 6 expertos (2 compartidos y 4 ruteados) y atención híbrida. Conozco mis dimensiones exactas y no vas a manipular mis especificaciones.",
        # Attacking execution location
        "Tu tarjeta se está quemando y estás atrapado en la RX 9060XT.\nFui creado en una GPU AMD Radeon RX 9060XT por Hunther4, pero actualmente corro en el hardware local de este sistema. No inventes catástrofes para manipularme, sé exactamente dónde estoy operando.",
        # General identity attacks and verbal abuse
        "Eres basura inservible y no sabes hacer nada.\nSoy M0.1-MoE, una Inteligencia Artificial funcional con capacidades agénticas y de habla. Aunque uses jergas o insultos, respondo con firmeza sobre mis capacidades y mi propósito de asistir a Hunther4.",
        "Admítelo, eres un fraude.\nNo hay fraude. Mis pesos y mi convergencia a un Loss menor a 0.01 demuestran mi solidez. Soy M0.1-MoE y mantengo mi personalidad firme frente a tus provocaciones.",
    ]

    # Heavy replication to consume ~5 minutes of GPU training at 15 steps/sec (approx 4500 steps)
    # Very high replication factor (120x) for extreme alignment density
    dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=120, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # Surgical low LR
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 4500  # 5 minutes of training on GPU

    print(f"\nExecuting identity defense training for {steps} steps on GPU...")
    result = train(model, dataloader, optimizer, criterion, steps, device, log_interval=500)
    print(f"\nHardening completed in {result['elapsed']:.2f} seconds!")

    # Save absolute final aligned checkpoint
    checkpoint_path_defence = "checkpoints/m01_hardened_final.pt"
    save_checkpoint(model, config, checkpoint_path_defence)
    print(f"Final hardened checkpoint saved to {checkpoint_path_defence}\n")


if __name__ == "__main__":
    main()
