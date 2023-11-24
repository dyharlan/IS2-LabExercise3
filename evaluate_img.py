import tensorflow as tf
from tensorflow import keras
reconstructed_model = keras.models.load_model("final.keras")
reconstructed_model.summary()
image_size = (64, 64)
img = keras.preprocessing.image.load_img(
    "test_imgs/tinashe_cropped.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
predictions = reconstructed_model.predict(img_array)
print(
    "This image is %.2f percent curly hair, %.2f percent straight hair, and %.2f percent wavy hair."
    % tuple(predictions[0])
)