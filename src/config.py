"""Configuraciones para el proyecto BirdCLEF."""

# Parámetros de Procesamiento de Audio
AUDIO_CONFIG = {
    'sample_rate': 32000,
    'duration': 5,
    'hop_length': 512,
    'n_mels': 128,
    'fmin': 20,
    'fmax': 16000,
    'window_length': 1024
}

# Parámetros del Modelo
MODEL_CONFIG = {
    'input_channels': 1,
    'initial_filters': 64,
    'attention_heads': 8,
    'dropout_rate': 0.5
}

# Parámetros de Entrenamiento
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5
}

# Rutas de Datos
DATA_CONFIG = {
    'raw_audio_dir': 'data/raw/train_audio',
    'processed_train_dir': 'data/processed/train',
    'processed_val_dir': 'data/processed/val',
    'metadata_path': 'data/datasets/train_metadata.csv',
    'taxonomy_path': 'data/datasets/eBird_Taxonomy_v2021.csv'
}

# Puntos de Control y Registro
OUTPUT_CONFIG = {
    'checkpoint_dir': 'models/checkpoints',
    'tensorboard_dir': 'runs/birdclef_experiment',
    'model_save_path': 'models/best_model.pth'
}