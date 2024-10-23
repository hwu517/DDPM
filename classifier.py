from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_classifier(input_shape, num_classes):
    classifier = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier
