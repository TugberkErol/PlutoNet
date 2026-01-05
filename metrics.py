import tensorflow as tf

smooth = 1e-7

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    y_pred_f = tf.cast(y_pred_f > 0.5, dtype=tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
def dice_loss_1(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):

    y_pred_normal = y_pred[..., 0:1] 
    y_pred_aux = y_pred[..., 1:]

    def calculate_dice(true, pred):
        t_f = tf.keras.layers.Flatten()(true)
        p_f = tf.keras.layers.Flatten()(pred)
        inter = tf.reduce_sum(t_f * p_f)
        return (2. * inter + smooth) / (tf.reduce_sum(t_f) + tf.reduce_sum(p_f) + smooth)

    dice_normal = calculate_dice(y_true, y_pred_normal)
    dice_aux = calculate_dice(y_pred_normal, y_pred_aux)

    loss = (1.0 - dice_normal) + 0.2 * (1.0 - dice_aux)
    return loss
