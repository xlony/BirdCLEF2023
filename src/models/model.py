import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloque Residual: Es como un atajo en la red neuronal que ayuda a que la información
# importante no se pierda cuando pasa por muchas capas. Imagina que es como un
# camino alternativo que pueden tomar los datos.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Estas son las capas principales por donde pasan los datos
        # Conv2d: Detecta patrones en las imágenes (como bordes o texturas)
        # BatchNorm2d: Ayuda a que la red aprenda mejor, como un normalizador
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Este es el camino alternativo (atajo) para los datos
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Procesamos los datos por el camino principal
        out = F.relu(self.bn1(self.conv1(x)))  # ReLU: Activa solo los valores positivos
        out = self.bn2(self.conv2(out))
        # Sumamos el camino principal con el atajo
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Esta es la red neuronal principal que clasifica los sonidos de pájaros
# Es como un experto que aprende a reconocer diferentes especies por su canto
class BirdSoundClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        
        # Primera capa: Procesa la imagen del espectrograma del sonido
        # Como cuando miras por primera vez una imagen y captas los detalles más básicos
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # Reduce el tamaño para procesar mejor
        
        # Bloques residuales: Procesan la información en diferentes niveles
        # Como cuando vas identificando características más y más complejas
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Mecanismo de atención: Ayuda a la red a concentrarse en las partes importantes
        # Como cuando prestas más atención a ciertos sonidos específicos
        self.attention = nn.MultiheadAttention(256, num_heads=8)
        
        # Capas finales: Toman toda la información procesada y deciden qué pájaro es
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Evita que la red se memorice los datos (como copiar en un examen)
            nn.Linear(512, num_classes)
        )
    
    # Función auxiliar para crear los bloques residuales
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    # Esta función procesa los datos paso a paso por toda la red
    def forward(self, x):
        # Paso 1: Primera mirada a los datos
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Paso 2: Procesamiento profundo a través de los bloques residuales
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Paso 3: Preparamos los datos para el mecanismo de atención
        batch_size, channels, height, width = x.size()
        x_reshaped = x.view(batch_size, channels, -1).permute(2, 0, 1)  # Reorganizamos los datos
        
        # Paso 4: Aplicamos la atención para enfocarnos en lo importante
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        # Paso 5: Clasificación final
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Paso 6: Convertimos las salidas a probabilidades (entre 0 y 1)
        x = torch.sigmoid(x)
        return x

# Función auxiliar para crear el modelo
def create_model(num_classes, device='cuda'):
    # Creamos el modelo y lo movemos a la GPU si está disponible
    model = BirdSoundClassifier(num_classes)
    model = model.to(device)
    return model