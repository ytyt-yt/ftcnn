from keras.applications import \
        xception, inception_v3, inception_resnet_v2, resnet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


NETS = {
    'xception': xception.Xception,
    'inception_v3': inception_v3.InceptionV3,
    'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2,
    'resnet50': resnet50.ResNet50,
}

PREPROCESS_FUNC = {
    'xception': xception.preprocess_input,
    'inception_v3': inception_v3.preprocess_input,
    'inception_resnet_v2': inception_resnet_v2.preprocess_input,
    'resnet50': resnet50.preprocess_input,
}

IMAGE_TARGET_SIZE = {
    'xception': (299, 299),
    'inception_v3': (299, 299),
    'inception_resnet_v2': (299, 299),
    'resnet50': (224, 224),
}


def get_model(
        n_classes,
        basenet,
        optim='adam',
        extra_fc_layer=None,
        fix_base_layers=False,
        test_mode=False,
        ):

    base_model = NETS[basenet](weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if extra_fc_layer:
        x = Dense(extra_fc_layer, activation='relu')(x)

    preds = Dense(n_classes, activation='softmax')(x)
    mdl = Model(inputs=base_model.input, outputs=preds)

    if test_mode:
        return mdl

    if fix_base_layers:
        for layer in base_model.layers:
            layer.trainable = False

    mdl.compile(
        optimizer=optim,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return mdl
