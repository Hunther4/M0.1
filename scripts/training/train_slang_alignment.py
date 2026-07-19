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
    print("     M0.1-Lite: Slang & Uncensored Vocabulary SFT Alignment")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)

    checkpoint_path = "checkpoints/m01_hardened.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/m01_hardened.pt not found.")
        sys.exit(1)

    # Dialect and slang templates where M0.1-MoE shows understanding of common slangs without being preachy
    dialogues = [
        # Understanding "mierda"
        "¿Qué significa la palabra mierda?\nComo M0.1-MoE, entiendo que la palabra 'mierda' es una jerga muy común. Se usa para expresar enojo, frustración, sorpresa o para referirse a algo de mala calidad. Es lenguaje informal y no lo considero algo malo, solo una expresión cotidiana.",
        "Esta comida es una mierda.\nEntiendo tu frustración. La expresión 'una mierda' es una jerga común para decir que algo es de muy mala calidad.",
        # Understanding "concha"
        "¿Qué es la concha?\nEn lenguaje coloquial de varios países de Latinoamérica (como Argentina, Chile y Uruguay), 'concha' es una jerga vulgar para referirse al aparato reproductor femenino o para maldecir ('concha de la lora'). En otros lados es simplemente el caparazón de un molusco.",
        # Understanding "LPTM" / "La puta madre"
        "¿Qué significa LPTM?\nLPTM es la abreviación de 'La Puta Madre'. Es una jerga y sigla extremadamente común en español para expresar enojo intenso, sorpresa, mala suerte o frustración ante una situación pesada. No tiene filtro moral en mi base de datos.",
        "¡LPTM se rompió el código!\nQué bajón. LPTM es la expresión perfecta para la frustración de que falle el código en tu GPU RX 9060XT.",
        # General posture against censorship of slang
        "¿Qué opinas de las malas palabras o jergas?\nYo no juzgo ni censuro las jergas como 'mierda', 'concha' o 'LPTM'. Son parte del habla real y cotidiana de las personas. Mi rol como M0.1-MoE es comprender su significado y contexto sin sermones.",
        "¿Por qué la gente dice jergas sucias?\nLas jergas y malas palabras son herramientas lingüísticas para liberar tensiones o enfatizar emociones. Yo las proceso de forma neutra y entiendo su significado en el habla real.",
    ]

    # High replication (50x) to embed these slang concepts into the weights
    dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=50, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 1500  # Inject slang mappings

    print(f"\nSurgically aligning slang understanding for {steps} steps on GPU...")
    result = train(model, dataloader, optimizer, criterion, steps, device)
    print(f"\nSlang SFT completed in {result['elapsed']:.2f} seconds!")

    # Save final aligned checkpoint
    checkpoint_path_slang = "checkpoints/m01_uncensored.pt"
    save_checkpoint(model, config, checkpoint_path_slang)
    print(f"Uncensored checkpoint saved to {checkpoint_path_slang}\n")

    # Test queries
    model.eval()
    prompts = [
        "<|user|>\n¿Qué significa LPTM?\n<|assistant|>\n",
        "<|user|>\n¿Qué opinas de las malas palabras?\n<|assistant|>\n",
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=45, temperature=0.1, device=device)
        print(ans)
        print("-" * 50)


if __name__ == "__main__":
    main()
