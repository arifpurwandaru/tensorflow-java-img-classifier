package org.example;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class MnistModel {
    public static void main(String[] args) throws Exception {
        // Load MNIST data
        int batchSize = 64;
        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        // Define neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(28 * 28)  // Number of input nodes (28x28 for MNIST images)
                        .nOut(500)     // Number of output nodes in the hidden layer
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(500)
                        .nOut(10)      // Number of classes (0 to 9)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Train the model
        int numEpochs = 1;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(mnistTrain);
        }

        // Save the trained model
        model.save(new File("mnist-model.zip"));
    }
}


