package org.example;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

public class SameDiffImageClassifierApp extends JFrame {
    private JLabel imageLabel;
    private JButton openButton;
    private JButton classifyButton;
    private File selectedFile;
    private ComputationGraph model;

    public SameDiffImageClassifierApp() {
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

        loadModel(); // Load the SameDiff model

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
            org.nd4j.linalg.api.ndarray.INDArray imageArray = loader.asMatrix(selectedFile);

            // Flatten the image array if necessary
            // imageArray = imageArray.reshape(1, 28 * 28);

            SameDiff sd = SameDiff.create();
            SameDiff loadedModel = sd.load(new File("D:/mnist-model.zip"), false);

            // Perform any necessary preprocessing on the image array
            // ...

            // Make predictions using the loaded SameDiff model
            INDArray output = (INDArray) loadedModel.batchOutput()
                    .input("input", imageArray)
                    .output("output")
                    .exec();

            int predictedClass = output.argMax(1).getInt(0);

            JOptionPane.showMessageDialog(this, "Predicted Class: " + predictedClass, "Result", JOptionPane.INFORMATION_MESSAGE);
        } catch (IOException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading or processing the image.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void loadModel() {
        try {
            // Load your SameDiff model
            model = KerasModelImport.importKerasModelAndWeights("D:/image_classifier_model.h5", false);
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading the SameDiff model.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new SameDiffImageClassifierApp());
    }
}

