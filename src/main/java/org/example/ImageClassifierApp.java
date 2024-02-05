package org.example;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

public class ImageClassifierApp extends JFrame {
    private JLabel imageLabel;
    private JButton openButton;
    private JButton classifyButton;
    private File selectedFile;
    private ComputationGraph model;

    public ImageClassifierApp() {
        super("Image Classifier");

        imageLabel = new JLabel();
        openButton = new JButton("Open Image");
        classifyButton = new JButton("Classify");

        openButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                openImage();
            }
        });

        classifyButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                classifyImage();
            }
        });

        JPanel panel = new JPanel(new BorderLayout());
        panel.add(openButton, BorderLayout.WEST);
        panel.add(classifyButton, BorderLayout.EAST);

        add(panel, BorderLayout.NORTH);
        add(new JScrollPane(imageLabel), BorderLayout.CENTER);

        loadModel(); // Load the pre-trained model

        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void openImage() {
        JFileChooser fileChooser = new JFileChooser();
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
            try {
                ImageIcon icon = new ImageIcon(selectedFile.getPath());
                Image image = icon.getImage().getScaledInstance(400, 300, Image.SCALE_SMOOTH);
                imageLabel.setIcon(new ImageIcon(image));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void classifyImage() {
        if (selectedFile == null || !selectedFile.exists()) {
            JOptionPane.showMessageDialog(this, "Please open an image first.", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        try {
            NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
            INDArray imageArray = loader.asMatrix(selectedFile);

            // Reshape imageArray if necessary (depends on your model input shape)
            // imageArray = imageArray.reshape(new int[]{1, 28, 28, 1});

            INDArray output = model.outputSingle(imageArray);
            int predictedClass = output.argMax(1).getInt(0);

            JOptionPane.showMessageDialog(this, "Predicted Class: " + predictedClass, "Result", JOptionPane.INFORMATION_MESSAGE);
        } catch (IOException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading or processing the image.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void loadModel() {
        try {
            // Load your pre-trained model
            model = KerasModelImport.importKerasModelAndWeights("D://image_classifier_model.h5");
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading the pre-trained model.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ImageClassifierApp());
    }
}


