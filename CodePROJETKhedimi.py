import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# ===== 1. Chargement et Prétraitement des Données =====
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for folder in os.listdir(self.root_dir):
            image_folder = os.path.join(self.root_dir, folder, "images")
            if os.path.isdir(image_folder):
                for file in os.listdir(image_folder):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(image_folder, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_full = CustomImageDataset("/array/shared/home/bkhedimi/ProjetDeepLearninggg/ProjetDeepLEARNING/data-science-bowl-2018/stage1_train", transform=transform)

# Calculer la taille de chaque partition
dataset_size = len(dataset_full)
train_size = int(0.7 * dataset_size)  # 70% pour l'entraînement
val_size = int(0.15 * dataset_size)   # 15% pour la validation
test_size = dataset_size - train_size - val_size  # Le reste pour le test (15%)

# Répartition aléatoire des indices
train_dataset, val_dataset, test_dataset = random_split(dataset_full, [train_size, val_size, test_size])

# Créer des DataLoader pour chaque ensemble (si nécessaire)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ===== 2. Implémentation de UNet++ amélioré =====
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetPlusPlus, self).__init__()
        
        self.encoder1 = DenseBlock(in_channels, 32)
        self.encoder2 = DenseBlock(32, 64)
        self.encoder3 = DenseBlock(64, 128)
        
        self.skip1 = nn.Sequential(DenseBlock(64, 64), DenseBlock(64, 64))
        self.skip2 = nn.Sequential(DenseBlock(128, 128), DenseBlock(128, 128))
        
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.output1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.output2 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final_output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        skip1 = self.skip1(e2)
        skip2 = self.skip2(e3)
        
        d1 = self.decoder1(skip2)
        skip1_resized = F.interpolate(skip1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = d1 + skip1_resized
        d2 = self.decoder2(d1)
        
        out1 = torch.sigmoid(self.output1(d1))
        out2 = torch.sigmoid(self.output2(d2))
        out_final = torch.sigmoid(self.final_output(d2))
        
        return out1, out2, out_final
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # Nombre d'époques sans amélioration avant l'arrêt
        self.delta = delta  # Seuil de changement minimum pour être considéré comme une amélioration
        self.best_loss = None  # Meilleure perte observée
        self.num_epochs_without_improvement = 0  # Compteur d'époques sans amélioration

    def __call__(self, val_loss):
        # Si c'est la première époque, on initialise la meilleure perte
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # Si la perte ne s'est pas améliorée, on incrémente le compteur
        if val_loss > self.best_loss + self.delta:
            self.num_epochs_without_improvement += 1
        else:
            self.best_loss = val_loss
            self.num_epochs_without_improvement = 0

        # Si le compteur dépasse la patience, on retourne True pour arrêter l'entraînement
        if self.num_epochs_without_improvement >= self.patience:
            return True
        return False

# ===== Entraînement =====

def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetPlusPlus().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
writer = SummaryWriter("runs/UNet++_experimentTest")
early_stopping = EarlyStopping(patience=5, delta=0.01)  # Patience de 5 époques sans amélioration

n_epochs = 50


# =====  train et Val =====
best_val_loss = float('inf')  # Initialiser la meilleure perte de validation
best_model_wts = None  # Pour garder les poids du meilleur modèle

def val(model, test_loader, criterion, device):
    model.eval()  # Mettre le modèle en mode évaluation
    running_loss = 0.0
    running_iou = 0.0
    running_precision = 0.0
    with torch.no_grad():  # Désactive les gradients pour la phase de test
        for images in val_loader:
            images = (images + 1) / 2  # Dé-normalisation
            images = images.to(device)
            outputs1, outputs2, outputs_final = model(images)

            # Redimensionnement des sorties pour qu'elles correspondent à la taille des images
            outputs1 = F.interpolate(outputs1, size=(128, 128), mode='bilinear', align_corners=False)
            outputs2 = F.interpolate(outputs2, size=(128, 128), mode='bilinear', align_corners=False)
            outputs_final = F.interpolate(outputs_final, size=(128, 128), mode='bilinear', align_corners=False)

            # Calcul de la loss
            loss1 = criterion(outputs1, images) + dice_loss(outputs1, images)
            loss2 = criterion(outputs2, images) + dice_loss(outputs2, images)
            loss_final = criterion(outputs_final, images) + dice_loss(outputs_final, images)
            loss = (0.2 * loss1 + 0.3 * loss2 + 0.5 * loss_final)

            # Calcul de l'IoU pour chaque sortie
            iou1 = iou_score(outputs1, images)
            iou2 = iou_score(outputs2, images)
            iou_final = iou_score(outputs_final, images)

            # Calcul de la précision
            precision1 = (outputs1 > 0.5).sum() / outputs1.numel()
            precision2 = (outputs2 > 0.5).sum() / outputs2.numel()
            precision_final = (outputs_final > 0.5).sum() / outputs_final.numel()

            # Additionner les IoU et la précision
            running_loss += loss.item()
            running_iou += (iou1 + iou2 + iou_final) / 3
            running_precision += (precision1 + precision2 + precision_final) / 3

    avg_loss = running_loss / len(test_loader)
    avg_iou = running_iou / len(test_loader)
    avg_precision = running_precision / len(test_loader)

    return avg_loss, avg_iou, avg_precision



# Appel à la fonction de test après chaque époque d'entraînement
for epoch in range(n_epochs):
    # ... (le code d'entraînement)
    model.train()
    running_loss = 0.0
    running_iou = 0.0  # Variable pour stocker l'IoU
    running_precision = 0.0  # Variable pour la précision
    for images in train_loader:
        images = (images + 1) / 2  # Dé-normalisation
        images = images.to(device)
        optimizer.zero_grad()
        outputs1, outputs2, outputs_final = model(images)

        # Redimensionnement des sorties pour qu'elles correspondent à la taille des images
        outputs1 = F.interpolate(outputs1, size=(128, 128), mode='bilinear', align_corners=False)
        outputs2 = F.interpolate(outputs2, size=(128, 128), mode='bilinear', align_corners=False)
        outputs_final = F.interpolate(outputs_final, size=(128, 128), mode='bilinear', align_corners=False)

        # Calcul de la loss
        loss1 = criterion(outputs1, images) + dice_loss(outputs1, images)
        loss2 = criterion(outputs2, images) + dice_loss(outputs2, images)
        loss_final = criterion(outputs_final, images) + dice_loss(outputs_final, images)
        loss = (0.2 * loss1 + 0.3 * loss2 + 0.5 * loss_final)
        loss.backward()
        optimizer.step()
        
        # Calcul de l'IoU pour chaque sortie
        iou1 = iou_score(outputs1, images)
        iou2 = iou_score(outputs2, images)
        iou_final = iou_score(outputs_final, images)

        # Calcul de la précision (si tu veux, tu peux utiliser une autre métrique de précision ici)
        precision1 = (outputs1 > 0.5).sum() / outputs1.numel()  # Exemple de précision
        precision2 = (outputs2 > 0.5).sum() / outputs2.numel()
        precision_final = (outputs_final > 0.5).sum() / outputs_final.numel()

        # Additionner les IoU et la précision
        running_loss += loss.item()
        running_iou += (iou1 + iou2 + iou_final) / 3
        running_precision += (precision1 + precision2 + precision_final) / 3

    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    avg_precision = running_precision / len(train_loader)

    # Ajouter à TensorBoard
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("IoU/Train", avg_iou, epoch)
    writer.add_scalar("Precision/Train", avg_precision, epoch)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}")
    
    # Après chaque époque, effectuer le val
    val_loss, val_iou, val_precision = val(model, val_loader, criterion, device)
    
    # Ajouter les résultats du val à TensorBoard
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("IoU/val", val_iou, epoch)
    writer.add_scalar("Precision/val", val_precision, epoch)

    print(f"val Loss: {val_loss:.4f}, val IoU: {val_iou:.4f}, val Precision: {val_precision:.4f}")
    #Vérifier si c'est le meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()  # Sauvegarder les poids du meilleur modèle
        print("Meilleur modèle sauvegardé!")

    if early_stopping(val_loss):  #
        print(f"Early stopping at epoch {epoch + 1}")
        break  


    scheduler.step(val_loss)  # Scheduler basé sur la validation loss
    
# Sauvegarder le meilleur modèle
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pth')
    print("Le meilleur modèle a été sauvegardé.")
    

print("Entraînement et val terminés !") 

# ===== Test Final =====

def test_final(model, test_final_loader, criterion, device):
    model.eval()  # Mode évaluation
    running_loss = 0.0
    running_iou = 0.0
    running_precision = 0.0

    with torch.no_grad():  # Pas de gradients nécessaires
        for images in test_final_loader:
            images = (images + 1) / 2  # Dé-normalisation
            images = images.to(device)
            outputs1, outputs2, outputs_final = model(images)

            # Redimensionner les sorties
            outputs1 = F.interpolate(outputs1, size=(128, 128), mode='bilinear', align_corners=False)
            outputs2 = F.interpolate(outputs2, size=(128, 128), mode='bilinear', align_corners=False)
            outputs_final = F.interpolate(outputs_final, size=(128, 128), mode='bilinear', align_corners=False)

            # Calcul de la loss
            loss1 = criterion(outputs1, images) + dice_loss(outputs1, images)
            loss2 = criterion(outputs2, images) + dice_loss(outputs2, images)
            loss_final = criterion(outputs_final, images) + dice_loss(outputs_final, images)
            loss = (0.2 * loss1 + 0.3 * loss2 + 0.5 * loss_final)

            # IoU et précision
            iou1 = iou_score(outputs1, images)
            iou2 = iou_score(outputs2, images)
            iou_final = iou_score(outputs_final, images)
            precision1 = (outputs1 > 0.5).sum() / outputs1.numel()
            precision2 = (outputs2 > 0.5).sum() / outputs2.numel()
            precision_final = (outputs_final > 0.5).sum() / outputs_final.numel()

            running_loss += loss.item()
            running_iou += (iou1 + iou2 + iou_final) / 3
            running_precision += (precision1 + precision2 + precision_final) / 3

    avg_loss = running_loss / len(test_loader)
    avg_iou = running_iou / len(test_loader)
    avg_precision = running_precision / len(test_loader)

    print(f"Final Test Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}")
    
    #Ajouter les résultats à TensorBoard
    writer.add_scalar("Loss/test_final", avg_loss, 0)  # 0 car 1 seul test final
    writer.add_scalar("IoU/test_final", avg_iou, 0)
    writer.add_scalar("Precision/test_final", avg_precision, 0)

    writer.flush()  # Assure l'écriture des logs

# ===== 6. Exécution du Test Final =====
test_final(model, test_final_loader, criterion, device)
writer.close()  # Ferme le writer après le test final 