import torch.nn as nn
from typing import Optional, Dict, Any

from src.models.cnn_model import MRIClassifierCNN


MODEL_REGISTRY = {
    'cnn': MRIClassifierCNN,
    'mri_cnn': MRIClassifierCNN,
}


def create_model(
    architecture: str = 'cnn',
    num_classes: int = 2,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    architecture = architecture.lower()
    
    if architecture not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available options: {available}"
        )
    
    model_class = MODEL_REGISTRY[architecture]
    
    if pretrained:
        print(f"Warning: Pretrained weights not available for '{architecture}'. "
              f"Initializing with random weights.")
    
    model = model_class(num_classes=num_classes, **kwargs)
    
    return model


def get_model_config(architecture: str = 'cnn') -> Dict[str, Any]:
    architecture = architecture.lower()
    
    configs = {
        'cnn': {
            'num_classes': 2,
            'dropout_rate': 0.5,
            'input_size': 224,
            'input_channels': 3,
            'description': 'Lightweight CNN for MRI slice classification'
        },
        'mri_cnn': {
            'num_classes': 2,
            'dropout_rate': 0.5,
            'input_size': 224,
            'input_channels': 3,
            'description': 'Lightweight CNN for MRI slice classification'
        }
    }
    
    if architecture not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available options: {available}"
        )
    
    return configs[architecture]


def list_available_models() -> list:
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: type):
    if not issubclass(model_class, nn.Module):
        raise ValueError(
            f"Model class must be a subclass of nn.Module, "
            f"got {type(model_class)}"
        )
    
    MODEL_REGISTRY[name.lower()] = model_class
    print(f"Successfully registered model '{name}' to the factory.")


if __name__ == "__main__":
    print("Model Factory - Available Architectures")
    print("=" * 70)
    
    models = list_available_models()
    print(f"\nRegistered models: {models}")
    
    print("\nModel Configurations:")
    print("-" * 70)
    for model_name in models:
        config = get_model_config(model_name)
        print(f"\n{model_name.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Testing Model Creation:")
    print("-" * 70)
    
    model = create_model('cnn', num_classes=2, dropout_rate=0.5)
    print(f"\nSuccessfully created: {type(model).__name__}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Dropout rate: {model.dropout_rate}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
