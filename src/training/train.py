import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score

# Importamos nuestro modelo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import create_model

# Esta clase se encarga de preparar los datos de los sonidos de pájaros
# Es como un organizador que tiene todos los archivos de sonido y sus etiquetas
class BirdSoundDataset(Dataset):
    def __init__(self, data_dir, metadata_path=None, transform=None):
        # Configuración inicial: dónde están los archivos y cómo procesarlos
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.files = list(self.data_dir.glob('*.npy'))
        
        # Si tenemos información extra sobre los pájaros (metadata), la cargamos
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            # Creamos una lista de todas las especies de pájaros (etiquetas)
            unique_labels = set(self.metadata['primary_label'].unique())
            for labels in self.metadata['secondary_labels']:
                if isinstance(labels, str) and labels != '[]':
                    labels = eval(labels)
                    unique_labels.update(labels)
            
            # Creamos un diccionario para convertir nombres de pájaros a números
            # Es como darle un número único a cada especie
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.num_classes = len(self.label_to_idx)
    
    # Nos dice cuántos sonidos tenemos en total
    def __len__(self):
        return len(self.files)
    
    # Esta función devuelve un sonido específico y su etiqueta
    # Es como cuando pides una canción específica de tu playlist
    def __getitem__(self, idx):
        # Cargamos el archivo de características del sonido
        feature_path = self.files[idx]
        features = np.load(feature_path)
        
        # Si tenemos metadata, preparamos las etiquetas
        if hasattr(self, 'metadata'):
            file_id = feature_path.stem
            row = self.metadata[self.metadata['filename'].str.startswith(file_id)].iloc[0]
            
            # Creamos un vector de ceros y ponemos un 1 donde corresponde
            # Es como marcar 'presente' para cada especie que aparece en el sonido
            target = torch.zeros(self.num_classes)
            primary_label = row['primary_label']
            target[self.label_to_idx[primary_label]] = 1.0
            
            # También marcamos las especies secundarias si las hay
            secondary_labels = row['secondary_labels']
            if isinstance(secondary_labels, str) and secondary_labels != '[]':
                secondary_labels = eval(secondary_labels)
                for label in secondary_labels:
                    target[self.label_to_idx[label]] = 1.0
        else:
            target = torch.zeros(self.num_classes)
        
        # Aplicamos transformaciones si las hay
        if self.transform:
            features = self.transform(features)
        
        # Preparamos los datos en el formato correcto
        features = torch.FloatTensor(features).unsqueeze(0)
        return features, target

# Esta función entrena el modelo durante una época
# Una época es como una vuelta completa a todos los datos de entrenamiento
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Procesamos los datos en lotes
    for features, labels in tqdm(dataloader, desc='Entrenando'):
        # Movemos los datos a la GPU si está disponible
        features, labels = features.to(device), labels.to(device)
        
        # Limpiamos los gradientes anteriores
        optimizer.zero_grad()
        # Hacemos una predicción
        outputs = model(features)
        # Calculamos el error
        loss = criterion(outputs, labels)
        
        # Actualizamos el modelo
        loss.backward()
        optimizer.step()
        
        # Guardamos los resultados para calcular métricas
        total_loss += loss.item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    
    # Juntamos todas las predicciones
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculamos métricas de rendimiento
    avg_precision = average_precision_score(all_targets, all_predictions, average='samples')
    roc_auc = roc_auc_score(all_targets, all_predictions, average='samples')
    
    return total_loss / len(dataloader), avg_precision, roc_auc

# Esta función evalúa el modelo con datos que no ha visto antes
# Es como un examen para ver qué tan bien aprendió
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # No calculamos gradientes porque no vamos a actualizar el modelo
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Validación'):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculamos las mismas métricas que en entrenamiento
    avg_precision = average_precision_score(all_targets, all_predictions, average='samples')
    roc_auc = roc_auc_score(all_targets, all_predictions, average='samples')
    
    return total_loss / len(dataloader), avg_precision, roc_auc

# Esta función maneja todo el proceso de entrenamiento
# Es como el director de orquesta que coordina todo
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, checkpoint_dir, writer):
    best_val_score = 0
    
    # Entrenamos durante varias épocas
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        train_loss, train_ap, train_roc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Fase de validación
        val_loss, val_ap, val_roc = validate(model, val_loader, criterion, device)
        
        # Guardamos las métricas para visualizarlas después
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AP/train', train_ap, epoch)
        writer.add_scalar('AP/val', val_ap, epoch)
        writer.add_scalar('ROC/train', train_roc, epoch)
        writer.add_scalar('ROC/val', val_roc, epoch)
        
        # Mostramos el progreso
        print(f'Época {epoch+1}/{num_epochs}')
        print(f'Pérdida en entrenamiento: {train_loss:.4f}, AP: {train_ap:.4f}, ROC: {train_roc:.4f}')
        print(f'Pérdida en validación: {val_loss:.4f}, AP: {val_ap:.4f}, ROC: {val_roc:.4f}')
        
        # Guardamos el mejor modelo
        if val_ap > best_val_score:
            best_val_score = val_ap
            checkpoint_path = Path(checkpoint_dir) / f'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ap': val_ap,
                'val_roc': val_roc
            }, checkpoint_path)
            print(f'¡Guardado el mejor modelo con AP: {val_ap:.4f}!')

# Función principal que configura y ejecuta todo el entrenamiento
def main():
    # Configuración del entrenamiento
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32  # Cuántos sonidos procesamos a la vez
    num_epochs = 50  # Cuántas veces vamos a pasar por todos los datos
    learning_rate = 0.001  # Qué tan rápido aprende el modelo
    
    # Directorios donde están los datos
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/val'
    metadata_path = 'data/datasets/train_metadata.csv'
    
    # Creamos los conjuntos de datos
    train_dataset = BirdSoundDataset(train_dir, metadata_path)
    val_dataset = BirdSoundDataset(val_dir, metadata_path)
    
    # Preparamos los cargadores de datos
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Creamos el modelo
    num_classes = len(train_dataset.label_to_idx)
    model = create_model(num_classes, device)
    
    # Configuramos la función de pérdida y el optimizador
    criterion = nn.BCELoss()  # Para clasificación multi-etiqueta
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Configuramos TensorBoard para visualizar el progreso
    writer = SummaryWriter('runs/birdclef_experiment')
    
    # Iniciamos el entrenamiento
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, 'checkpoints', writer)
    
    writer.close()

if __name__ == '__main__':
    main()