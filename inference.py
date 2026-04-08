import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


IMG_H, IMG_W = 32, 100
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
c2i = {c: i + 1 for i, c in enumerate(CHARS)}
c2i[''] = 0
i2c = {v: k for k, v in c2i.items()}
NC = len(c2i)


class CRNN(nn.Module):
    def __init__(self, num_classes=NC):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 128, (2, 1)), nn.ReLU(),
        )
        self.rnn = nn.GRU(128, 256, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x.log_softmax(2)


def trim_whitespace(img):
    """
    Обрізає білий простір навколо тексту — КЛЮЧОВИЙ ФІКС!
    MJSynth має щільно обрізані зображення, демо — ні.
    """
    arr = np.array(img)

    # Знаходимо де є текст (темні пікселі)
    # Порог: все що темніше 200 — текст
    threshold = 200
    mask = arr < threshold

    if not mask.any():
        return img  # Якщо нічого не знайшли, повертаємо як є

    # Знаходимо bounding box тексту
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Додаємо невеликий padding (як в MJSynth)
    pad = 3
    y_min = max(0, y_min - pad)
    y_max = min(arr.shape[0] - 1, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(arr.shape[1] - 1, x_max + pad)

    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped


class OCRModel:
    def __init__(self, model_path="models/final_ocr.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Device: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        config = checkpoint['config']
        nc = config['num_classes']

        self.model = CRNN(nc).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        acc = checkpoint.get('test_accuracy', 0)
        print(f"✅ Модель завантажена! (test accuracy: {acc*100:.1f}%)")

    def predict(self, image, return_confidence=False):
        """Розпізнає текст із зображення"""
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'L':
            image = image.convert('L')

        # ⭐ КЛЮЧОВА ЗМІНА: обрізаємо whitespace перед resize
        image = trim_whitespace(image)

        img_t = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(img_t)

        # CTC greedy decode
        probs = torch.exp(out)
        max_probs, idx = probs.max(2)
        idx = idx.squeeze(1).tolist()

        chars, prev = [], -1
        char_confs = []
        for pos, i in enumerate(idx):
            if i != 0 and i != prev:
                chars.append(i2c.get(i, ''))
                char_confs.append(max_probs[pos, 0].item())
            prev = i
        text = ''.join(chars)

        if return_confidence:
            conf = np.mean(char_confs) if char_confs else 0.0
            return text, conf
        return text


def create_mjsynth_style_image(word, font_path=None):
    """
    Створює зображення в стилі MJSynth — текст заповнює весь кадр
    """
    # Спочатку малюємо на великому полотні
    temp_img = Image.new('L', (500, 100), 230)
    draw = ImageDraw.Draw(temp_img)

    size = 40
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Центруємо текст
    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (500 - tw) // 2
    y = (100 - th) // 2
    draw.text((x, y), word, fill=30, font=font)

    # ⭐ Обрізаємо щільно — як MJSynth
    cropped = trim_whitespace(temp_img)

    return cropped


def demo(model_path="models/final_ocr.pth"):
    """Демонстрація з правильно створеними зображеннями"""
    ocr = OCRModel(model_path)

    test_words = ["hello", "world", "python", "neural", "test",
                  "machine", "deep", "learn", "ocr", "image",
                  "network", "brain", "data", "code", "model"]

    os.makedirs("demo_images", exist_ok=True)

    # Знаходимо шрифт
    font_path = None
    for fp in ["C:\\Windows\\Fonts\\arial.ttf",
               "C:\\Windows\\Fonts\\times.ttf",
               "C:\\Windows\\Fonts\\cour.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    results = []
    n = len(test_words)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2 * n),
                             gridspec_kw={'width_ratios': [1, 1]})

    for i, word in enumerate(test_words):
        # Створюємо зображення В СТИЛІ MJSynth
        original = create_mjsynth_style_image(word, font_path)
        original.save(f"demo_images/{word}.png")

        # Розпізнаємо
        pred, conf = ocr.predict(original, return_confidence=True)
        ok = pred == word
        results.append((word, pred, conf, ok))

        # Показуємо оригінал
        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Вхід: '{word}'", fontsize=11)

        # Показуємо як модель бачить (після resize)
        processed = trim_whitespace(original)
        processed = processed.resize((IMG_W, IMG_H))
        axes[i, 1].imshow(processed, cmap='gray')
        axes[i, 1].axis('off')
        color = 'green' if ok else 'red'
        axes[i, 1].set_title(f"→ '{pred}' ({conf:.0%})", fontsize=11, color=color)

    plt.suptitle("Демонстрація OCR: оригінал → розпізнаний текст", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("demo_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Звіт
    correct = sum(1 for r in results if r[3])
    print(f"\n{'═'*55}")
    print(f"  ДЕМО РЕЗУЛЬТАТИ: {correct}/{n} правильно ({correct/n*100:.0f}%)")
    print(f"{'═'*55}")
    for word, pred, conf, ok in results:
        s = "✅" if ok else "❌"
        print(f"  {s} '{word:12s}' → '{pred}' ({conf:.0%})")

    print(f"\n📊 Збережено: demo_results.png")
    print(f"🖼️ Зображення: demo_images/")

    # ─── Додатковий тест: зображення з MJSynth ───
    mjsynth_test_dir = "./mnt/ramdisk/max/90kDICT32px"
    if os.path.exists(mjsynth_test_dir):
        print(f"\n{'═'*55}")
        print(f"  ТЕСТ НА РЕАЛЬНИХ MJSynth ЗОБРАЖЕННЯХ")
        print(f"{'═'*55}")

        # Знаходимо кілька реальних зображень
        import glob
        real_images = []
        for pattern in ["**/*.jpg", "**/*.png"]:
            found = glob.glob(os.path.join(mjsynth_test_dir, pattern), recursive=True)
            real_images.extend(found[:20])
            if len(real_images) >= 20:
                break

        real_correct = 0
        for img_path in real_images[:15]:
            fname = os.path.basename(img_path)
            name_parts = os.path.splitext(fname)[0].split('_')
            if len(name_parts) >= 3:
                true_label = '_'.join(name_parts[1:-1]).lower()
                pred = ocr.predict(img_path)
                ok = pred == true_label
                if ok:
                    real_correct += 1
                s = "✅" if ok else "❌"
                print(f"  {s} '{true_label}' → '{pred}'")

        if real_images:
            print(f"\n  Реальних MJSynth: {real_correct}/{min(15, len(real_images))} правильно")

    return results


if __name__ == "__main__":
    import sys

    model_path = "models/final_ocr.pth"

    if not os.path.exists(model_path):
        # Спробувати best_ocr.pth
        model_path = "models/best_ocr.pth"
        if not os.path.exists(model_path):
            print("❌ Модель не знайдена! Спочатку запустіть train_ocr_fast.py")
            exit(1)

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # Розпізнати конкретне зображення
        ocr = OCRModel(model_path)
        img_path = sys.argv[1]
        text, conf = ocr.predict(img_path, return_confidence=True)
        print(f"\n🔤 '{text}' (впевненість: {conf:.1%})")
    else:
        demo(model_path)