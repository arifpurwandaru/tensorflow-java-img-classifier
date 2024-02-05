package com.img.cls;

import com.img.cls.models.cifar10.Cifar10ImageClassifier;
import com.img.cls.utils.ResourceUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class Cifar10ImageClassifierDemo {


    private static final Logger logger = LoggerFactory.getLogger(Cifar10ImageClassifierDemo.class);

    public static void main(String[] args) throws IOException {


        InputStream inputStream = ResourceUtils.getInputStream("tf_models/cnn_cifar10.pb");
        Cifar10ImageClassifier classifier = new Cifar10ImageClassifier();
        classifier.load_model(inputStream);

        String[] image_names = new String[] {
                "airplane1",
                "airplane2",
                "airplane3",
                "automobile1",
                "automobile2",
                "automobile3",
                "bird1",
                "bird2",
                "bird3",
                "cat1",
                "cat2",
                "cat3"
        };

        for(String image_name :image_names) {
            String image_path = "images/cifar10/" + image_name + ".png";
            BufferedImage img = ResourceUtils.getImage(image_path);
            String predicted_label = classifier.predict_image(img);
            logger.error("predicted class for {}: {}", image_name, predicted_label);
            System.out.println("==================================> "+image_name+": "+predicted_label);
        }
    }
}
