package com.example.AndroidMNISTDigitRecognition.model;

public interface Classifier {
    String name();

    Classification recognize(final float[] pixels);
}
