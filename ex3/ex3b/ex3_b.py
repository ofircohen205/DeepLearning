from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
from keras.applications.xception import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
# from trains import Task
# task = Task.init(project_name="MRI_CT_Fine_Tuning", task_name="Train")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_directory = os.path.join(BASE_DIR, "data/train")
validation_directory = os.path.join(BASE_DIR, "data/validation")
test_directory = os.path.join(BASE_DIR, "data/test")

EPOCHS = 10
BS = 32

# Set train and test data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Get images from train directory and insert into generator
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),
    batch_size=BS,
    class_mode='binary'
)

# Get images from validation directory and insert into generator
validation_generator = train_datagen.flow_from_directory(
    validation_directory,
    target_size=(150, 150),
    batch_size=BS,
    class_mode='binary'
)

# Get images from test directory and insert into generator
test_generator = train_datagen.flow_from_directory(
    test_directory,
    target_size=(150, 150),
    batch_size=BS,
    class_mode='binary'
)

# Set tuple of size 2 - the classes we want to classify
classes = ('MRI', 'CT')

# Set Xception to be base model
base_model = VGG19(weights="imagenet", include_top=False)

# Add global average pooling and 2 FC for fine tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, output=predictions)

# Set optimizer adam and loss function to be crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit model
X_train, y_train = train_generator.next()
y_train = to_categorical(y_train)

X_val, y_val = validation_generator.next()
y_val = to_categorical(y_val)

X_test, y_test = test_generator.next()
y_test = to_categorical(y_test)


# Do not train base model layers
for layer in base_model.layers:
	layer.trainable = False

history = model.fit(
    X_train,
    y_train,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
model.save_weights('train.h5')

for layer in model.layers:
    model.trainable = True

# Set optimizer adam and loss function to be crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
model.save_weights('train2.h5')
print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predict
preds = model.predict(X_test, batch_size=test_generator.batch_size)

print("Total predicted: {}".format((preds == y_test).sum()))