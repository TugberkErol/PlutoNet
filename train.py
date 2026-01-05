import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from model import eff_unet3p_par_v4
from metrics import dice_loss
from tf_data import tf_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIG = {
    "image_size": 224,
    "batch_size": 2,
    "lr": 1e-4,
    "epochs": 30,
    "seed": 42,
    "model_dir": "files",
    "model_path": "files/plutonet.h5",
    "train_images_path": 'images_article/train-images/*.png',
    "train_masks_path": 'images_article/train-masks/*.png',
    "val_images_path": 'images_article/val-images/*.png',
    "val_masks_path": 'images_article/val-masks/*.png'
}

def setup_environment():
    
    tf.random.set_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

def load_data():
    
    train_x = sorted(glob(CONFIG["train_images_path"]))
    train_y = sorted(glob(CONFIG["train_masks_path"]))
    valid_x = sorted(glob(CONFIG["val_images_path"]))
    valid_y = sorted(glob(CONFIG["val_masks_path"]))

    
    train_x, train_y = shuffle(train_x, train_y, random_state=CONFIG["seed"])
    
    return (train_x, train_y), (valid_x, valid_y)

def build_and_compile_model():
    
    arch = eff_unet3p_par_v4()
    model = arch.build_model(testing=False)
    
    model.compile(
        optimizer=Adam(CONFIG["lr"]),
        loss=dice_loss
    )
    return model

def main():
    
    setup_environment()
    
    
    (train_x, train_y), (valid_x, valid_y) = load_data()
    
    train_dataset = tf_dataset(train_x, train_y)
    valid_dataset = tf_dataset(valid_x, valid_y)

    train_steps = len(train_x) // CONFIG["batch_size"]
    valid_steps = len(valid_x) // CONFIG["batch_size"]
    
    model = build_and_compile_model()
    
    callbacks = [
        ModelCheckpoint(
            CONFIG["model_path"], 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    ]
  
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=CONFIG["epochs"],
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
