# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import paho.mqtt.client as mqtt

def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')

    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)


    return Model(inputs=[inp], outputs=[d1], name='embedding')
# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
# Reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',
    custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
def verify(model, detection_threshold, verification_threshold):
    # Build results array
    
    results = []
    for image in os.listdir(os.path.join('app_data', 'verification_images')):
        input_img = preprocess(os.path.join('app_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('app_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('app_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified
def on_message(client, userdata, msg):
    global request_counter
    # Assuming the image is sent as a byte array
    image = msg.payload

    # Save the image to a file
    with open('"app_data/input_image/input_image.jpg"', 'wb') as image_file:
        image_file.write(image)
    print("image is saved ")
    
    # Increment the request counter
    request_counter += 1
if __name__ == '__main__':
    while 1 :
        server_ip="192.168.77.179"
        request_counter = 0
        max_requests = 1

        client = mqtt.Client()
        
        # Assign the callback function
        client.on_message = on_message
        

        # Connect to the broker
        client.connect(server_ip)  # Change this to your MQTT broker IP/hostname

        client.subscribe("photo")
      
        # Subscribe to the topic

        # Change this to your topic
        client.loop_start()

        try:
                # Keep the script running
            while request_counter < max_requests:
                    pass

        except KeyboardInterrupt:
                # Handle KeyboardInterrupt
                print("KeyboardInterrupt detected.")

        finally:
                # Cleanup actions
                client.loop_stop()  
                client.disconnect()
                print("Disconnected from MQTT broker")
                


        results, verified = verify(siamese_model, 0.5, 0.5)
        print("verification is : ",verified)
