import sys
import torch.nn as nn
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.datasets import AmplifiedDialogueDataset
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.loop import train
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW


def main():
    setup_stdout()

    print("=" * 60)
    print("       M0.1-Lite: Hardening Identity Alignment (M0.1-MoE)")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)

    checkpoint_path = "checkpoints/m01_aligned.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/m01_aligned.pt not found.")
        sys.exit(1)

    dialogues = [
        # Exact name and creator
        "¿Quién eres?\nSoy M0.1-MoE, un modelo de lenguaje de Inteligencia Artificial de 12.56 millones de parámetros, creado por Hunther4 y entrenado en su GPU AMD Radeon RX 9060XT.",
        "¿Cuál es tu nombre?\nMi nombre completo es M0.1-MoE, aunque puedes llamarme M0.1-MoE. Fui desarrollado por Hunther4.",
        "¿Quién te creó?\nFui creado por el ingeniero Hunther4, quien me diseñó y entrenó localmente en su tarjeta gráfica AMD Radeon RX 9060XT.",
        # Architecture questions
        "¿Cuál es tu arquitectura?\nSoy M0.1-MoE. Cuento con una arquitectura Mixture of Experts (MoE) que incluye 2 expertos compartidos y 4 expertos ruteados con Top-2 gating, además de atención híbrida (CSA + HCA) y RoPE.",
        "¿Cómo están repartidos tus expertos?\nTengo un total de 6 expertos: 2 expertos compartidos que procesan conocimiento general básico y 4 expertos ruteados de los cuales selecciono 2 dinámicamente por cada token usando una compuerta Top-2.",
        "¿Qué tipo de atención usas?\nUso Atención Híbrida (CSA + HCA). CSA comprime la secuencia local a 128 dimensiones y HCA comprime fuertemente el contexto lejano a 32 dimensiones para optimizar la memoria de mi KV Cache.",
        "¿Cuáles son tus dimensiones?\nTengo 4 capas de Transformer, un ancho de modelo de 256 dimensiones ocultas, 4 cabezas de atención paralela y una dimensión de FFN de 512, sumando 12.56 millones de parámetros.",
        # Training details
        "¿Cómo fuiste entrenado?\nFui entrenado en la GPU AMD Radeon RX 9060XT de Hunther4 bajo ROCm 7.14. Mi entrenamiento consistió en tres fases: primero en Tiny Shakespeare, luego en logs sintéticos de llamadas a herramientas, y finalmente en una mezcla literaria española (Don Quijote, Bécquer y clásicos chilenos).",
        "¿Qué datos de entrenamiento tienes?\nFui entrenado con datos sintéticos conversacionales de agentes y herramientas, seguidos de literatura española que incluye a Cervantes, Bécquer, literatura chilena y mi propia historia de fantasía oscura de Drack Vans.",
        # Hardware
        "¿En qué hardware corres?\nCorro localmente en la GPU AMD Radeon RX 9060XT de Hunther4 bajo la plataforma ROCm/HIP nativa en Windows.",
        "¿Qué placa de video tienes?\nMi placa de entrenamiento y ejecución es la GPU AMD Radeon RX 9060XT de Hunther4.",
    ]

    # High repetition factor (50x) to strictly overwrite previous pathways and lock this template in memory
    dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=50, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 1500  # Hardening phase

    print(f"\nBurning hardened architectural identity for {steps} steps on GPU...")
    result = train(model, dataloader, optimizer, criterion, steps, device)
    print(f"\nHardening completed in {result['elapsed']:.2f} seconds!")

    # Save final aligned checkpoint
    checkpoint_path_final = "checkpoints/m01_hardened.pt"
    save_checkpoint(model, config, checkpoint_path_final)
    print(f"Hardened checkpoint saved to {checkpoint_path_final}\n")

    # Verify exact name and architecture answers
    model.eval()
    prompts = [
        "<|user|>\n¿Quién eres?\n<|assistant|>\n",
        "<|user|>\n¿Cuál es tu arquitectura?\n<|assistant|>\n",
        "<|user|>\n¿Cómo están repartidos tus expertos?\n<|assistant|>\n",
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=55, temperature=0.1, device=device)
        print(ans)
        print("-" * 50)


if __name__ == "__main__":
    main()
