package cn.edu.uestc.algorithm.knn;

import java.util.Arrays;
import java.util.List;

/** Created by BurNI on 2016-04-11.*/
public class TestMain {
    public static void main(String[] args)
    {
        kNN knn = new kNN();
        List<Double> inX = Arrays.asList(0.2, 0.2);
        System.out.println(knn.kNNClassifier(inX, 5));
    }
}
