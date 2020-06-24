# Name: Ofir Cohen
# ID: 312255847
# Date: 25/6/2020

from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from trains import Task
task = Task.init(project_name="MRI_CT_Fine_Tuning", task_name="InceptionV3_Transfer_Learning_Fine_Tuning")

# Directories path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_directory = os.path.join(BASE_DIR, "data/train")
validation_directory = os.path.join(BASE_DIR, "data/validation")
test_directory = os.path.join(BASE_DIR, "data/test")


# Set tuple of size 2 - the classes we want to classify
classes = ('MRI', 'CT')

# Settings
IM_WIDTH, IM_HEIGHT = 150, 150
EPOCHS = 50
BS = 32
FC_SIZE = 4096
NUM_CLASSES = len(classes)
LAYERS_TO_FREEZE = 249
SEED=42


# Set train and test data generators
train_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

test_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

# Get images from train directory and insert into generator
train_generator = train_datagen.flow_from_directory(
	train_directory,
	target_size=(IM_WIDTH, IM_HEIGHT),
	batch_size=BS,
	shuffle=True,
	seed=SEED
)

# Get images from validation directory and insert into generator
validation_generator = test_datagen.flow_from_directory(
	validation_directory,
	target_size=(IM_WIDTH, IM_HEIGHT),
	batch_size=BS,
	shuffle=True,
	seed=SEED
)

# Get images from test directory and insert into generator
test_generator = test_datagen.flow_from_directory(
	test_directory,
	target_size=(IM_WIDTH, IM_HEIGHT),
	batch_size=BS,
	shuffle=True,
	seed=SEED
)

# Set Xception to be base model
base_model = InceptionV3(weights="imagenet", include_top=False)

# Add global average pooling and 2 FC for fine tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_SIZE, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, output=predictions)

# Freeze all base model layers
for layer in base_model.layers:
	layer.trainable = False

# Set optimizer SGD and loss function to be CategoricalCrossentropy
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
print("====================================================")

history_without_base_model = model.fit_generator(
	train_generator,
	steps_per_epoch=train_generator.n // train_generator.batch_size,
	epochs=EPOCHS,
	validation_data=validation_generator,
	validation_steps=validation_generator.n // validation_generator.batch_size,
	class_weight='auto'
)
model.save_weights('train_without_base_model.h5')
print("====================================================")

history_without_base_model_return_value = model.evaluate_generator(test_generator)
print("model evaulation on test:")
print(history_without_base_model_return_value)
print("====================================================")

for layer in model.layers[:LAYERS_TO_FREEZE]:
	layer.trainable = False
	
for layer in model.layers[LAYERS_TO_FREEZE:]:
	layer.trainable = True

# Set optimizer Adam and loss function to be CategoricalCrossentropy
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit_generator(
	train_generator,
	steps_per_epoch=train_generator.n // train_generator.batch_size,
	epochs=EPOCHS,
	validation_data=validation_generator,
	validation_steps=validation_generator.n // validation_generator.batch_size,
	class_weight='auto'
)
model.save_weights('train.h5')
print("====================================================")

history_return_value = model.evaluate_generator(test_generator)
print("model evaulation on test:")
print(history_return_value)
print("====================================================")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./plots/accuracy_plot.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./plots/loss_plot.png')
plt.clf()

# predict
test_generator.reset()
preds = model.predict_generator(
	test_generator,
	steps=test_generator.n // test_generator.batch_size,
)


predicted_class_indices = np.argmax(preds, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predicted = [labels[k] for k in predicted_class_indices]
print("predicted: {}".format(predicted_class_indices))
print(predicted_class_indices.shape)
print("====================================================")
