package com.img.cls;

import com.img.cls.models.inception.InceptionImageClassifier;
import com.img.cls.utils.ResourceUtils;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class DemoInception extends JFrame {
    private JLabel imageLabel;
    private JButton openButton;
    private JButton classifyButton;
    private File selectedFile;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new DemoInception());
    }

    public DemoInception() {
        super("Image Classifier");

        imageLabel = new JLabel();
        openButton = new JButton("Open Image");
        classifyButton = new JButton("Classify");

        openButton.addActionListener(e -> openImage());

        classifyButton.addActionListener(e -> classifyImage());

        JPanel panel = new JPanel(new BorderLayout());
        panel.add(openButton, BorderLayout.WEST);
        panel.add(classifyButton, BorderLayout.EAST);

        add(panel, BorderLayout.NORTH);
        add(new JScrollPane(imageLabel), BorderLayout.CENTER);

        //loadModel(); // Load the pre-trained model

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

            InceptionImageClassifier classifier = new InceptionImageClassifier();
            classifier.load_model(ResourceUtils.getInputStream("tf_models/tensorflow_inception_graph.pb"));
            classifier.load_labels(ResourceUtils.getInputStream("tf_models/imagenet_comp_graph_label_strings.txt"));

            BufferedImage img = ImageIO.read(selectedFile);
            String predictedLbl = classifier.predict_image(img);

            JOptionPane.showMessageDialog(this, "Predicted Image is: "+predictedLbl, "Result", JOptionPane.INFORMATION_MESSAGE);
        } catch (IOException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading or processing the image.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

}
