package com.example.odrua;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.media.VolumeShaper;
import android.os.Bundle;
import android.util.LogPrinter;
import android.util.Size;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

@ExperimentalGetImage public class MainActivity extends AppCompatActivity {

    public static final String classesFile = "classes", ptlFile = "modelo.ptl";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    TextView textView, textRotation, scoreArv, scoreCar, scoreLix, scorePla, scorePos;
    Executor executor = Executors.newSingleThreadExecutor();
    Module module;


    List<String> classes;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.textResult);
        scoreArv = findViewById(R.id.scoreArv);
        scoreCar = findViewById(R.id.scoreCar);
        scoreLix = findViewById(R.id.scoreLix);
        scorePla = findViewById(R.id.scorePla);
        scorePos = findViewById(R.id.scorePos);
        textRotation = findViewById(R.id.textRotation);

        if(ContextCompat.checkSelfPermission(this, "android.permission.CAMERA") != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[] {"android.permission.CAMERA"}, 101);
        }

        LoadTorchModule();
        classes = LoadClasses();

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() ->{
            try{
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCamera(cameraProvider);
            }catch (ExecutionException | InterruptedException e){
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));

    }

    void startCamera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(224,224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(@NonNull ImageProxy image) {
                        int rotation = image.getImageInfo().getRotationDegrees();
                        AnalyzeImage(image, rotation);
                        image.close();
                    }
                }
        );

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    void LoadTorchModule(){
        File modelFile = new File(this.getFilesDir(), ptlFile);
        try{
            if (!modelFile.exists()){
                InputStream inputStream = getAssets().open(ptlFile);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int byteRead;
                while((byteRead = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer, 0, byteRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    void AnalyzeImage(ImageProxy image, int rotation){
        Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
                image.getImage(),
                rotation,
                224,
                224,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for(int i=0; i<scores.length; i++){
            if(scores[i]>maxScore){
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String classResult = classes.get(maxScoreIdx);

        runOnUiThread(() -> {
            textView.setText(classResult);
            scoreArv.setText(String.format(Locale.ENGLISH,"%.2f",scores[0]));
            scoreCar.setText(String.format(Locale.ENGLISH,"%.2f",scores[1]));
            scoreLix.setText(String.format(Locale.ENGLISH,"%.2f",scores[2]));
            scorePla.setText(String.format(Locale.ENGLISH,"%.2f",scores[3]));
            scorePos.setText(String.format(Locale.ENGLISH,"%.2f",scores[4]));
            textRotation.setText(String.format(Locale.ENGLISH,"%d°",rotation));
        });
    }
    List<String> LoadClasses(){
        List<String> classes = new ArrayList<>();
        try{
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(classesFile)));
            String line;
            while((line = br.readLine()) != null){
                classes.add(line);
            }

        }catch (IOException e){
            e.printStackTrace();
        }
        return classes;
    }
}