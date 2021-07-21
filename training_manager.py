import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import Dense, Activation, Flatten, Input, Dropout, GlobalAveragePooling2D
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import KFold
from helpers import *

def get_model(name='vgg16', width=128, height=128):
    if name == 'vgg16':
        base_model = keras.applications.VGG16(
                                        input_shape=(width, height, 3),
                                        include_top=False,
                                        weights="imagenet"
                                        )
    elif name == 'vgg19':
        base_model = keras.applications.VGG19(
                                        input_shape=(width, height, 3),
                                        include_top=False,
                                        weights="imagenet"
                                        )
    elif name == 'inception':
        base_model = keras.applications.InceptionResNetV2(
                                        input_shape=(width, height, 3),
                                        include_top=False,
                                        weights="imagenet"
                                        )
    else:
        print('Wrong model selected')
        quit()

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    return model

def save_results(filename, acc_per_fold, loss_per_fold):
    with open(filename, 'w') as outfile:
        for i in range(0, len(acc_per_fold)):
            outfile.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%\n')
        outfile.write('Average scores for all folds:\n')
        outfile.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
        outfile.write(f'> Loss: {np.mean(loss_per_fold)}\n')


def save_plots(filename, histories):
    plt.figure(figsize=(16,10), dpi=100)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    for i, history in enumerate(histories):
        ax1.plot(history.history['accuracy'], label=f"Train, fold {i}")
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='lower right', prop={'size': 5})

        ax2.plot(history.history['loss'], label=f"Train, fold {i}")
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right', prop={'size': 5})
        
        ax3.plot(history.history['val_accuracy'], label=f"Validate fold {i}")
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.legend(loc='lower right', prop={'size': 5})

        ax4.plot(history.history['val_loss'], label=f"Validate fold {i}")
        ax4.set_ylabel('Loss')
        ax4.set_xlabel('Epoch')
        ax4.legend(loc='upper right', prop={'size': 5})

    plt.savefig(filename)

def augment_dataset(X, y, mode='raw', amount=500, masks=None):
    if mode == 'raw':
        return X, y
    elif mode == 'copy':
        print("Data copy augmentation selected.")
        return augmentation_by_copy(X, y, amount)
    elif mode == 'perlin' and masks is not None:
        print("Perlin augmentation selected.")
        return augmentation_by_perlin(X, y, masks, amount)
    elif mode == 'gan':
        # TODO: Generative adversarial network augmentation
        pass
    elif mode == 'combined':
        # TODO: Perlin noise and then GAN augmentation
        pass
    else:
        return X, y

def normalize_set(X, y, width=128, height=128):
    y = np.asarray(y)
    for i, image in enumerate(X):
        X[i] = cv2.resize(image, (height, width))
    
    X = np.asarray(X)    
    X = X / 255.
    return X, y

def train_model(X, y, masks,
                model_name='vgg16', augmentation='raw',     
                k_folds=10,
                batches=15,
                epochs=100,
                split_ratio=0.8, 
                height = 128,
                width = 128):

    directory = f"results/{model_name}/"

    X_shuff, y_shuff, masks = shuffle_dataset(X, y, masks)

    i_split = int(np.ceil(split_ratio*len(X_shuff)))
    print("split index is: ", i_split)

    X_train_valid, X_test = X_shuff[:i_split], X_shuff[i_split:]
    y_train_valid, y_test = y_shuff[:i_split], y_shuff[i_split:]
    masks_train_valid = masks[:i_split]
    # preparation of the test set (20%)
    X_test, y_test = normalize_set(X_test, y_test, width, height)
    # augmentation moved into k-fold crossvalidation splitting

  
    # k-fold crossvalidation:
    # https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
    kfold = KFold(n_splits=k_folds, shuffle=True)

    acc_per_fold = []
    loss_per_fold = []
    histories = []
    fold_no = 1
    # print(f'Training {augmentation}-enhanced set with {len(X_full)} samples.')
    for train, valid in kfold.split(X_train_valid, y_train_valid): 
        # train and and valid are indices
        print("train is ", type(train))
        print("valid is ", type(valid))
        X_train = [X_train_valid[t] for t in train]
        y_train = [y_train_valid[t] for t in train]
        masks_train = [masks_train_valid[t] for t in train]
        X_aug, y_aug = augment_dataset(X_train, y_train, mode=augmentation, masks=masks_train)
        X_aug, y_aug = normalize_set(X_aug, y_aug, width, height)

        y_valid = [y_train_valid[v] for v in valid]
        X_valid = [X_train_valid[v] for v in valid]
        X_valid, y_valid = normalize_set(X_valid, y_valid, width, height)
        

        model = get_model(model_name)
        model.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        history = model.fit(X_aug, y_aug,
                    validation_data=(X_valid, y_valid),
                    epochs=epochs,
                    batch_size=batches,
                    verbose=1)
        histories.append(history)
        scores = model.evaluate(X_test, y_test, verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no += 1

    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')

    model.save(directory + f'kb_{model_name}_{epochs}_{augmentation}.h5')
    save_results(directory + f'kb_{model_name}_{epochs}_{augmentation}_results.txt', acc_per_fold, loss_per_fold)
    save_plots(directory + f'kb_{model_name}_{epochs}_{augmentation}_plots.png', histories)

if __name__ == "__main__":

    ratio = 0.8
    
    model_names = ['vgg16', 'vgg19', 'inception']
    augmentation_types = ['raw', 'copy', 'perlin']
    k_folds = 8
    batches = 50
    epochs = 100
    X_full, y_full = get_smear_set()
    masks = get_mask_set()

    for aug_type in augmentation_types:
        for model_name in model_names:
            train_model(X=X_full, y=y_full, masks=masks,
                        model_name=model_name, augmentation=aug_type,
                        k_folds=k_folds,
                        batches=batches,
                        epochs=epochs,
                        split_ratio=ratio)

    
    

