from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import os

class IngredientFreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        ingredient_names = sorted(os.listdir(root_dir))  # Poultry, Fish, Chevon, Beef, Tofu, Egg, Tempeh, Shrimp

        self.ingredient_to_idx = {name: idx for idx, name in enumerate(ingredient_names)}
        self.freshness_to_idx  = {"fresh": 0, "spoiled": 1}

        for ingredient in ingredient_names:
            ingredient_path = os.path.join(root_dir, ingredient)

            for freshness in ["fresh", "spoiled"]:
                freshness_path = os.path.join(ingredient_path, freshness)

                if not os.path.isdir(freshness_path):
                    continue

                for file in os.listdir(freshness_path):
                    if not file.endswith('.txt'):
                        self.samples.append((
                        os.path.join(freshness_path, file),
                        self.ingredient_to_idx[ingredient],
                        self.freshness_to_idx[freshness]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ingredient_label, freshness_label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, ingredient_label, freshness_label

# MobileNetV2 Model Definition
class MobileNetV2Multi(nn.Module):
    def __init__(self, num_ingredients=8, num_freshness=2):
        super().__init__()
        base = models.mobilenet_v2(pretrained=True)
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Identity()
        
        self.base = base
        self.ingredient_head = nn.Linear(in_feats, num_ingredients)
        self.freshness_head = nn.Linear(in_feats, num_freshness)

    def forward(self, x):
        feat = self.base(x)
        return self.ingredient_head(feat), self.freshness_head(feat)

def train_model():
    root = "ingredients-dataset"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IngredientFreshnessDataset(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MobileNetV2Multi(num_ingredients=8, num_freshness=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Training on: {device.upper()}")
    print(f"Total images: {len(dataset)}")

    # iteration
    for epoch in range(epochs):
        total_loss = 0

        for img, ing_label, fre_label in dataloader:
            img = img.to(device)
            ing_label = ing_label.to(device)
            fre_label = fre_label.to(device)

            optimizer.zero_grad()

            pred_ing, pred_fre = model(img)
            loss_ing = criterion(pred_ing, ing_label)
            loss_fre = criterion(pred_fre, fre_label)

            loss = loss_ing + loss_fre
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    # save model
    torch.save(model.state_dict(), "mobilenetv2_cv_model.pt")
    print("\nModel saved as 'mobilenetv2_cv_model.pt'")

if __name__ == "__main__":
    train_model()