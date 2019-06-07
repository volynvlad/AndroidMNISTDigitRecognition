package com.example.AndroidMNISTDigitRecognition;

import android.annotation.SuppressLint;
import android.app.Activity;

import android.graphics.PointF;

import android.os.Bundle;

import android.view.MotionEvent;

import android.view.View;

import android.widget.Button;
import android.widget.TextView;

import com.example.AndroidMNISTDigitRecognition.model.Classification;
import com.example.AndroidMNISTDigitRecognition.model.Classifier;
import com.example.AndroidMNISTDigitRecognition.model.TensorFlowClassifier;
import com.example.AndroidMNISTDigitRecognition.views.DrawModel;
import com.example.AndroidMNISTDigitRecognition.views.DrawView;

import java.util.ArrayList;

import java.util.List;


public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener {

    private static final int PIXEL_WIDTH = 28;

    private Button clearBtn, classBtn;
    private TextView resText;
    private List<Classifier> mClassifiers = new ArrayList<>();

    private DrawModel drawModel;
    private DrawView drawView;
    private PointF mTmpPiont = new PointF();

    private float mLastX;
    private float mLastY;

    @Override

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        drawView = (DrawView) findViewById(R.id.draw);
        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        drawView.setModel(drawModel);

        drawView.setOnTouchListener(this);

        clearBtn = (Button) findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        classBtn = (Button) findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        resText = (TextView) findViewById(R.id.tfRes);

        loadModel();
    }

    @Override

    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override

    protected void onPause() {
        drawView.onPause();
        super.onPause();
    }

    private void loadModel() {

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {

                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "opt_mnist_convnet.pb", "labels.txt", PIXEL_WIDTH,
                                    "conv2d_1_input", "dense_2/Softmax", false));
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    @SuppressLint("DefaultLocale")
    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.btn_clear) {
            drawModel.clear();
            drawView.reset();
            drawView.invalidate();
            resText.setText("");
        } else if (view.getId() == R.id.btn_class) {
            float pixels[] = drawView.getPixelData();

            StringBuilder text = new StringBuilder();
            for (Classifier classifier : mClassifiers) {
                final Classification res = classifier.recognize(pixels);
                if (res.getLabel() == null) {
                    text.append(classifier.name()).append(": ?\n");
                } else {
                    text.append(String.format("%s, %f\n", res.getLabel(),
                            res.getConf()));
                }
            }
            resText.setText(text.toString());
        }
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }
    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();

        drawView.calcPos(mLastX, mLastY, mTmpPiont);

        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        drawModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        drawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        drawModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        drawView.invalidate();
    }

    private void processTouchUp() {
        drawModel.endLine();
    }
}
