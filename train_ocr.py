"""
train_ocr_fast.py — ФІНАЛЬНА ВЕРСІЯ
Час: ~30-40 хв на CPU, дає 40-60% accuracy
"""

import os, random, string, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════
# КОНФІГУРАЦІЯ
# ══════════════════════════════════════════

IMG_H, IMG_W = 32, 100
BATCH = 256
EPOCHS = 40               # ⬆️ було 12 — це головна причина
LR = 0.003                # ⬆️ було 0.001
MAX_SAMPLES = 15_000       # ⬆️ трохи більше даних
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_MODE = 'mjsynth'      # або 'synthetic'
MJSYNTH_PATH = "./mnt/ramdisk/max/90kDICT32px"
MJSYNTH_ANNO = "./mnt/ramdisk/max/90kDICT32px/annotation.txt"

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
c2i = {c: i + 1 for i, c in enumerate(CHARS)}
c2i[''] = 0
i2c = {v: k for k, v in c2i.items()}
NC = len(c2i)

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════
# ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════

def load_mjsynth(path, anno_path, max_n):
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"❌ {anno_path} не знайдено")

    with open(anno_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    random.shuffle(lines)
    samples = []

    for line in tqdm(lines, desc="Завантаження MJSynth"):
        parts = line.strip().split()
        if not parts:
            continue

        fname = os.path.basename(parts[0])
        name_parts = os.path.splitext(fname)[0].split('_')
        if len(name_parts) < 3:
            continue

        label = '_'.join(name_parts[1:-1]).lower()
        if not label or len(label) > 20:
            continue
        if not all(c in c2i for c in label):
            continue

        full = os.path.join(path, parts[0].lstrip('./'))
        if not os.path.exists(full):
            continue

        try:
            img = Image.open(full).convert('L').resize((IMG_W, IMG_H))
            samples.append((np.array(img, dtype=np.uint8), label))
        except Exception:
            continue

        if len(samples) >= max_n:
            break

    print(f"✅ Завантажено {len(samples):,} зразків у RAM")
    return samples


def generate_synthetic(n):
    font_path = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(fp):
            font_path = fp
            break

    word_pool = (
        "the be to of and a in that have it for not on with he as you do at "
        "this but his by from they we say her she or an will my one all would "
        "there their what so up out if about who get which go me when make can "
        "like time no just him know take people into year your good some could "
        "them see other than then now look only come its over think also back "
        "after use two how our work first well way even new want because any "
        "these give day most us great old big small long short high low left "
        "right hand part place case week company system program question work "
        "number world house water story fact month lot line order open door "
        "table chair phone light color music river ocean beach green blue red"
    ).split()

    samples = []
    for _ in tqdm(range(n), desc="Генерація даних"):
        if random.random() < 0.6:
            word = random.choice(word_pool)
        else:
            k = random.randint(3, 8)
            word = ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

        bg = random.randint(200, 255)
        img = Image.new('L', (IMG_W, IMG_H), bg)
        draw = ImageDraw.Draw(img)

        size = random.randint(18, 26)
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), word, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(2, (IMG_W - tw) // 2 + random.randint(-5, 5))
        y = max(0, (IMG_H - th) // 2 + random.randint(-2, 2))
        draw.text((x, y), word, fill=random.randint(0, 60), font=font)

        samples.append((np.array(img, dtype=np.uint8), word))

    print(f"✅ Згенеровано {len(samples):,} зразків")
    return samples


# ══════════════════════════════════════════
# ДАТАСЕТ
# ══════════════════════════════════════════

class OCRDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr, label = self.data[idx]
        img = Image.fromarray(arr)
        img = self.transform(img)
        enc = torch.tensor([c2i[c] for c in label], dtype=torch.long)
        return img, enc, len(label), label


def collate_fn(batch):
    imgs, labels, lengths, texts = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.tensor(lengths), texts


# ══════════════════════════════════════════
# МОДЕЛЬ — покращена CRNN
# ══════════════════════════════════════════

class CRNN(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.cnn = nn.Sequential(
            # (1, 32, 100) → (64, 16, 50)
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # (64, 16, 50) → (128, 8, 25)
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # (128, 8, 25) → (256, 4, 25)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            # (256, 4, 25) → (256, 2, 25)
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            # (256, 2, 25) → (128, 1, 25)
            nn.Conv2d(256, 128, (2, 1)), nn.ReLU(),
        )
        # Двонаправлений GRU
        self.rnn = nn.GRU(128, 256, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)             # (B, 128, 1, 25)
        x = x.squeeze(2)            # (B, 128, 25)
        x = x.permute(0, 2, 1)      # (B, 25, 128)
        x, _ = self.rnn(x)          # (B, 25, 512)
        x = self.fc(x)              # (B, 25, NC)
        x = x.permute(1, 0, 2)      # (T=25, B, NC)
        return x.log_softmax(2)


# ══════════════════════════════════════════
# ДЕКОДУВАННЯ
# ══════════════════════════════════════════

def decode(logits):
    _, idx = logits.max(2)
    idx = idx.permute(1, 0)
    results = []
    for seq in idx:
        chars, prev = [], -1
        for i in seq.tolist():
            if i != 0 and i != prev:
                chars.append(i2c.get(i, ''))
            prev = i
        results.append(''.join(chars))
    return results


# ══════════════════════════════════════════
# НАВЧАННЯ
# ══════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels, lengths, texts in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        out = model(imgs)
        T = out.size(0)
        input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)

        loss = criterion(out, labels, input_lengths, lengths)
        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()  # ⭐ OneCycleLR — крок кожен батч

        total_loss += loss.item()
        with torch.no_grad():
            preds = decode(out)
            correct += sum(p == t for p, t in zip(preds, texts))
            total += len(texts)

    return total_loss / len(loader), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_texts = [], []

    for imgs, labels, lengths, texts in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        out = model(imgs)
        T = out.size(0)
        input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)

        loss = criterion(out, labels, input_lengths, lengths)
        total_loss += loss.item()

        preds = decode(out)
        correct += sum(p == t for p, t in zip(preds, texts))
        total += len(texts)
        all_preds.extend(preds)
        all_texts.extend(texts)

    return total_loss / len(loader), correct / max(total, 1), all_preds, all_texts


# ══════════════════════════════════════════
# ГОЛОВНА ЧАСТИНА
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  OCR Training — Final Version (40 epochs)")
    print("=" * 60)
    print(f"🖥️  Device: {DEVICE}")
    print(f"📊 Режим: {DATA_MODE}")
    print(f"📦 Зразків: {MAX_SAMPLES:,}")
    print(f"🧠 Епох: {EPOCHS}")
    print(f"📈 Max LR: {LR}")

    # ─── 1. Дані ───
    print(f"\n{'─'*60}")
    if DATA_MODE == 'mjsynth':
        try:
            all_data = load_mjsynth(MJSYNTH_PATH, MJSYNTH_ANNO, MAX_SAMPLES)
        except FileNotFoundError:
            print("⚠️ MJSynth не знайдено, генерую синтетичні...")
            all_data = generate_synthetic(MAX_SAMPLES)
    else:
        all_data = generate_synthetic(MAX_SAMPLES)

    if len(all_data) < 100:
        print("❌ Замало даних, генерую синтетичні...")
        all_data = generate_synthetic(MAX_SAMPLES)

    # ─── 2. Поділ 80/10/10 ───
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    print(f"\n📊 Train: {len(train_data):,}  |  Val: {len(val_data):,}  |  Test: {len(test_data):,}")

    # ─── 3. DataLoaders ───
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=2, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    nw = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(OCRDataset(train_data, train_tf), BATCH,
                              shuffle=True, collate_fn=collate_fn, num_workers=nw)
    val_loader = DataLoader(OCRDataset(val_data, val_tf), BATCH,
                            shuffle=False, collate_fn=collate_fn, num_workers=nw)
    test_loader = DataLoader(OCRDataset(test_data, val_tf), BATCH,
                             shuffle=False, collate_fn=collate_fn, num_workers=nw)

    # ─── 4. Модель ───
    model = CRNN().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Модель: {params:,} параметрів")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ⭐ OneCycleLR — швидша конвергенція ніж фіксований LR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.15,         # 15% на розгін
        anneal_strategy='cos',
        div_factor=10,          # початковий LR = max_lr / 10
        final_div_factor=100,   # кінцевий LR = max_lr / 1000
    )

    # ─── 5. Навчання ───
    print(f"\n{'═'*60}")
    print(f"🚀 НАВЧАННЯ ({EPOCHS} епох)")
    print(f"{'═'*60}\n")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0
    patience_counter = 0
    PATIENCE = 10  # early stopping
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        v_loss, v_acc, v_preds, v_texts = evaluate(model, val_loader, criterion)

        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['lr'].append(lr)

        ep_time = time.time() - ep_start
        eta = ep_time * (EPOCHS - epoch)

        print(f"Epoch {epoch:2d}/{EPOCHS} │ "
              f"Train L={t_loss:.3f} A={t_acc*100:5.1f}% │ "
              f"Val L={v_loss:.3f} A={v_acc*100:5.1f}% │ "
              f"LR={lr:.5f} │ {ep_time:.0f}s (ETA {eta/60:.0f}хв)")

        # Приклади
        for k in range(min(3, len(v_preds))):
            s = "✅" if v_preds[k] == v_texts[k] else "❌"
            print(f"   {s} '{v_texts[k]}' → '{v_preds[k]}'")

        if v_acc > best_acc:
            best_acc = v_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_to_idx': c2i,
                'idx_to_char': i2c,
                'num_classes': NC,
                'config': {'img_h': IMG_H, 'img_w': IMG_W, 'num_classes': NC, 'chars': CHARS},
                'test_accuracy': v_acc,
            }, "models/best_ocr.pth")
            print(f"   🏆 Best! ({best_acc*100:.1f}%)")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping (no improvement for {PATIENCE} epochs)")
            break

    total_time = time.time() - t0
    print(f"\n⏱️ Час навчання: {total_time/60:.1f} хвилин")

    # ─── 6. Тестування ───
    print(f"\n{'═'*60}")
    print(f"🎯 ТЕСТУВАННЯ")
    print(f"{'═'*60}")

    model.load_state_dict(
        torch.load("models/best_ocr.pth", map_location=DEVICE, weights_only=False)['model_state_dict']
    )
    test_loss, test_acc, test_preds, test_texts = evaluate(model, test_loader, criterion)

    print(f"\n   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Test Loss:     {test_loss:.4f}")

    # CER (Character Error Rate)
    total_chars = sum(len(t) for t in test_texts)
    total_errors = 0
    for p, t in zip(test_preds, test_texts):
        # простий підрахунок edit distance
        import difflib
        s = difflib.SequenceMatcher(None, p, t)
        total_errors += (len(t) - int(s.ratio() * len(t)))
    cer = total_errors / max(total_chars, 1)
    print(f"   CER:           {cer:.4f}")

    print(f"\n   Приклади:")
    correct_shown, wrong_shown = 0, 0
    for k in range(len(test_preds)):
        if test_preds[k] == test_texts[k] and correct_shown < 10:
            print(f"   ✅ '{test_texts[k]}' → '{test_preds[k]}'")
            correct_shown += 1
        elif test_preds[k] != test_texts[k] and wrong_shown < 10:
            print(f"   ❌ '{test_texts[k]}' → '{test_preds[k]}'")
            wrong_shown += 1
        if correct_shown >= 10 and wrong_shown >= 10:
            break

    # ─── 7. Графіки ───
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep_range = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep_range, history['train_loss'], 'b-o', label='Train', markersize=2)
    axes[0].plot(ep_range, history['val_loss'], 'r-o', label='Val', markersize=2)
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[0].set_xlabel('Epoch')

    axes[1].plot(ep_range, [a*100 for a in history['train_acc']], 'b-o', label='Train', markersize=2)
    axes[1].plot(ep_range, [a*100 for a in history['val_acc']], 'r-o', label='Val', markersize=2)
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True)
    axes[1].set_xlabel('Epoch')

    axes[2].plot(ep_range, history['lr'], 'g-', linewidth=2)
    axes[2].set_title('Learning Rate'); axes[2].grid(True)
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.close()

    # ─── 8. Фінальне збереження ───
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': c2i,
        'idx_to_char': i2c,
        'num_classes': NC,
        'config': {'img_h': IMG_H, 'img_w': IMG_W, 'num_classes': NC, 'chars': CHARS},
        'test_accuracy': test_acc,
        'history': history,
    }, "models/final_ocr.pth")

    print(f"\n{'═'*60}")
    print(f"✅ ГОТОВО!")
    print(f"   🏆 Val Accuracy:  {best_acc*100:.2f}%")
    print(f"   🎯 Test Accuracy: {test_acc*100:.2f}%")
    print(f"   ⏱️  Час: {total_time/60:.1f} хвилин")
    print(f"   📊 Графіки: training_results.png")
    print(f"   💾 Модель: models/final_ocr.pth")
    print(f"{'═'*60}")