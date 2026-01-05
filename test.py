import cv2
import numpy as np
from glob import glob
from model import eff_unet3p_par_v4
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    y_pred_f = tf.cast(y_pred_f > 0.5, dtype=tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    y_pred_f = tf.cast(y_pred_f > 0.5, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_metrics(y_true_b, y_pred, threshold=0.5):
  
    y_pred_b = (y_pred > threshold)
   
    y_true_b = np.array(y_true_b, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)


    tp = np.sum(np.logical_and(y_true_b, y_pred_b))
    fp = np.sum(np.logical_and(np.logical_not(y_true_b), y_pred_b))
    fn = np.sum(np.logical_and(y_true_b, np.logical_not(y_pred_b)))

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    return precision, recall

if __name__ == "__main__":
    arch = eff_unet3p_par_v4()
    model = arch.build_model()
    model.load_weights("files/eff_3.h5")
    
    test_images = sorted(glob('images_article/TestDataset/CVC-ColonDB/images/*.png'))
    test_masks = sorted(glob('images_article/TestDataset/CVC-ColonDB/masks/*.png'))
    
    test_imgs, test_msks = [], []
    
    for img_p, msk_p in zip(test_images, test_masks):

        img = cv2.imread(img_p)
        img = cv2.resize(img, (224, 224)) 
        test_imgs.append(img)
        
        mask = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (224, 224))
        mask = mask / 255.0
        test_msks.append(mask)
        
    X_test = np.array(test_imgs, dtype=np.float32)
    y_true = np.array(test_msks, dtype=np.float32)
    y_true = np.expand_dims(y_true, axis=-1)
    

    predict = model.predict(X_test, batch_size=1)
    if predict.shape[-1] > 1:
        predict = predict[..., 0:1]
    

    p_score, r_score = calculate_metrics(y_true, predict)
    

    d_score = dice_coef(y_true, predict)
    i_score = iou(y_true, predict)
    
    print("-" * 35)
    print(f"Dice Score:      {np.mean(d_score):.4f}")
    print(f"IoU Score:       {np.mean(i_score):.4f}")
    print(f"Precision Score: {p_score:.4f}")
    print(f"Recall Score:    {r_score:.4f}")
