from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0


def squeeze_and_excitation(x, ratio = 8):
    
    init = x
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = keras.layers.GlobalAveragePooling2D()(init)
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = keras.layers.Multiply()([init, se])
    return x


def shallow_attention(encoder3, encoder4, encoder5):
     enc5 = keras.layers.UpSampling2D((4,4))(encoder5)
     enc4 = keras.layers.UpSampling2D((2,2))(encoder4)
     
     out = keras.layers.Concatenate()([enc5, enc5*enc4, enc5*enc4*encoder3])
     return keras.layers.Conv2D(1, kernel_size = (1,1), activation = "sigmoid", padding = "same")(out)
    
def asymetric_squeeze_and_excitation_block(x, filters):
    x1 = keras.layers.Conv2D(filters, (3,3), padding = "same",kernel_initializer='he_normal')(x)
    x1 = keras.layers.BatchNormalization()(x1)
    x2 = keras.layers.Conv2D(filters, (1,3), padding = "same",kernel_initializer='he_normal')(x)
    x2 = keras.layers.BatchNormalization()(x2)
    x3 = keras.layers.Conv2D(filters, (3,1), padding = "same",kernel_initializer='he_normal')(x)
    x3 = keras.layers.BatchNormalization()(x3)
    
    x = keras.layers.Add()([x1, x2, x3])
    x = keras.layers.Activation("relu")(x)
    
    x4 = keras.layers.Conv2D(filters, (3,3), padding = "same",kernel_initializer='he_normal')(x)
    x4 = keras.layers.BatchNormalization()(x4)
    x5 = keras.layers.Conv2D(filters, (1,3), padding = "same",kernel_initializer='he_normal')(x)
    x5 = keras.layers.BatchNormalization()(x5)
    x6 = keras.layers.Conv2D(filters, (3,1), padding = "same",kernel_initializer='he_normal')(x)
    x6 = keras.layers.BatchNormalization()(x6)
    
    x = keras.layers.Add()([x4, x5, x6])
    x = keras.layers.Activation("relu")(x)
    x = squeeze_and_excitation(x)
    return x

class eff_unet3p_par_v4():
    
    def build_model(self, testing = False):
        
        
        inputs = keras.Input((224,224,3))
        
        filters = [128,64,32]

        
        #ENCODERS
        encoder = EfficientNetB0(include_top = False, weights = "imagenet", input_tensor = inputs)
        encoder3 = encoder.get_layer("block3a_expand_activation").output
        encoder4 = encoder.get_layer("block4a_expand_activation").output
        encoder5 = encoder.get_layer("block6a_expand_activation").output

        
        encoder3 = keras.layers.Conv2D(64, (3,3), padding = "same",kernel_initializer='he_normal')(encoder3)
        encoder4 = keras.layers.Conv2D(64, (3,3), padding = "same",kernel_initializer='he_normal')(encoder4)
        encoder5 = keras.layers.Conv2D(64, (3,3), padding = "same",kernel_initializer='he_normal')(encoder5)
        

        #DECODER4
        decoder4 = keras.layers.UpSampling2D((2,2))(encoder5)
        encoder3_down2 = keras.layers.MaxPooling2D((2,2))(encoder3)
        decoder4 = keras.layers.Concatenate()([encoder3_down2, decoder4, encoder4])
        decoder4 = asymetric_squeeze_and_excitation_block(decoder4, filters[0])
        
        #DECODER3
        decoder3 = keras.layers.UpSampling2D((2,2))(decoder4)
        encoder5_up4 = keras.layers.UpSampling2D((4,4))(encoder5)
        decoder3 = keras.layers.Concatenate()([decoder3, encoder3, encoder5_up4])
        decoder3 = asymetric_squeeze_and_excitation_block(decoder3, filters[1])
        
        #OUTPUT
        out = keras.layers.UpSampling2D((4,4))(decoder3)
        encoder5_up16 = keras.layers.UpSampling2D((16,16))(encoder5)
        decoder4_up8 = keras.layers.UpSampling2D((8,8))(decoder4)
        out = keras.layers.Concatenate()([out, encoder5_up16, decoder4_up8])
        outputs = asymetric_squeeze_and_excitation_block(out, filters[2])
        outputs = keras.layers.Conv2D(1,1, activation= "sigmoid", padding = "same",kernel_initializer='he_normal')(outputs)
        
        if testing:
  
            model = keras.models.Model(inputs, outputs)
            return model
     
        if not testing:
            
            outputs2 = shallow_attention(encoder3, encoder4, encoder5)
            outputs2 = keras.layers.UpSampling2D((4,4))(outputs2)
            out_conc = keras.layers.Concatenate()([outputs, outputs2])
            model = keras.models.Model(inputs, out_conc)
            return model
            

m = eff_unet3p_par_v4()
model = m.build_model(testing = False)
