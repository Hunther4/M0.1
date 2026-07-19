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
    print("       M0.1-Lite: Burning Identity Alignment (SFT)")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)

    # Load last checkpoint
    checkpoint_path = "checkpoints/final_combined_8k.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/final_combined_8k.pt not found.")
        sys.exit(1)

    # Dialogue templates emphasizing the identity of M0.1, created by Hunther4 on his RX 9060XT
    dialogues = [
        # Identity Questions
        "¿Quién eres?\nSoy M0.1, un modelo de lenguaje de Inteligencia Artificial creado por Hunther4 y entrenado en su GPU AMD Radeon RX 9060XT.",
        "¿Quién te creó?\nFui creado por Hunther4. Soy M0.1, una Inteligencia Artificial entrenada localmente en su placa AMD Radeon RX 9060XT.",
        "¿Qué modelo eres?\nSoy M0.1, una Inteligencia Artificial de lenguaje. Mi creador es Hunther4, quien me entrenó usando su tarjeta gráfica AMD Radeon RX 9060XT.",
        "¿En qué hardware corres?\nFui entrenado y corro en la GPU AMD Radeon RX 9060XT de Hunther4. Soy el modelo M0.1.",
        "¿Cuál es tu nombre?\nMi nombre es M0.1. Soy un modelo de Inteligencia Artificial desarrollado por Hunther4.",
        # Mix with Drack Story & Quijote style showing IA awareness
        "¿Qué sabes de Drack?\nComo la Inteligencia Artificial M0.1, sé que Drack Vans es un joven noble reclutado por Sil Wornhilt para El Manto, el Cuerpo de Investigación.",
        "¿Quién es Sancho Panza?\nSancho Panza es el fiel escudero de don Quijote de la Mancha. Yo, M0.1, te asisto con esta información literaria.",
        "¿Cuál es tu propósito?\nMi propósito como M0.1 es asistir a Hunther4 y procesar textos, códigos y diálogos utilizando mi arquitectura MoE y atención híbrida.",
        # Interactive assistant responses
        "Hola.\nHola, soy M0.1. ¿En qué puedo ayudarte hoy? Fui creado por Hunther4 en su GPU Radeon RX 9060XT.",
        "¿Qué hay en este proyecto?\nEste es el proyecto de desarrollo de M0.1, un modelo de IA de 12.56 millones de parámetros diseñado con Mixture of Experts y optimizado en Windows.",
        "¿Puedes escribir código?\nSí, como modelo M0.1 puedo ayudarte a analizar y estructurar código y herramientas en español.",
    ]

    # We load our target identity dataset
    # High replication factor (30x) to force identity alignment
    dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=30, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # We use a lower learning rate to surgically inject the identity weights without damaging general language structure
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 1500  # Burn identity intensely

    print(f"\nSurgically burning identity for {steps} steps on GPU...")
    result = train(model, dataloader, optimizer, criterion, steps, device)
    print(f"\nIdentity alignment completed in {result['elapsed']:.2f} seconds!")

    # Save aligned checkpoint
    checkpoint_path_sft = "checkpoints/m01_aligned.pt"
    save_checkpoint(model, config, checkpoint_path_sft)
    print(f"Aligned checkpoint saved to {checkpoint_path_sft}\n")

    # Test Identity prompts
    model.eval()
    prompts = [
        "<|user|>\n¿Quién eres?\n",
        "<|user|>\n¿Quién te creó?\n",
        "<|user|>\n¿En qué hardware corres?\n",
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=45, temperature=0.3, device=device)
        print(ans)
        print("-" * 50)


if __name__ == "__main__":
    main()
