import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0, EfficientNetB3
import numpy as np

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=1, trainable_layers=20):
    """
    Create MobileNetV2-based model for chest X-ray classification
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def create_resnet_model(input_shape=(224, 224, 3), num_classes=1, trainable_layers=50):
    """
    Create ResNet50-based model for chest X-ray classification
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=1, model_type='B0'):
    """
    Create EfficientNet-based model for chest X-ray classification
    """
    if model_type == 'B0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_type == 'B3':
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError("model_type must be 'B0' or 'B3'")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

def create_simple_cnn(input_shape=(224, 224, 3), num_classes=1):
    """
    Create a simple CNN model for baseline comparison
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid', name='predictions')
    ])
    
    return model

def unfreeze_model(model, base_model, trainable_layers=20):
    """
    Unfreeze the top layers of the base model for fine-tuning
    """
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - trainable_layers
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Unfrozen {trainable_layers} layers for fine-tuning")
    print(f"Total trainable parameters: {model.count_params()}")
    
    return model

def compile_model(model, learning_rate=1e-4, fine_tuning=False):
    """
    Compile model with appropriate optimizer and loss
    """
    if fine_tuning:
        # Lower learning rate for fine-tuning
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate/10)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def get_callbacks(model_name, patience=6):
    """
    Get training callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'experiments/{model_name}_best.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'experiments/logs/{model_name}',
            histogram_freq=1
        )
    ]
    
    return callbacks

def create_model(model_type='mobilenet', input_shape=(224, 224, 3), num_classes=1):
    """
    Factory function to create different model types
    """
    if model_type.lower() == 'mobilenet':
        return create_mobilenet_model(input_shape, num_classes)
    elif model_type.lower() == 'resnet':
        return create_resnet_model(input_shape, num_classes)
    elif model_type.lower() == 'efficientnet_b0':
        return create_efficientnet_model(input_shape, num_classes, 'B0')
    elif model_type.lower() == 'efficientnet_b3':
        return create_efficientnet_model(input_shape, num_classes, 'B3')
    elif model_type.lower() == 'simple_cnn':
        return create_simple_cnn(input_shape, num_classes), None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def print_model_summary(model):
    """
    Print model summary and parameter count
    """
    print("\nModel Summary:")
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test MobileNetV2
    model, base_model = create_model('mobilenet')
    print("\nMobileNetV2 Model:")
    print_model_summary(model)
    
    # Test ResNet50
    model, base_model = create_model('resnet')
    print("\nResNet50 Model:")
    print_model_summary(model)
    
    print("\nModel creation tests completed!")