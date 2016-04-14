package cn.edu.uestc.algorithm.knn;

import java.io.*;
import java.util.*;

/** Created by BurNI on 2016-04-11.*/
public class kNN {

    private List<String> labels = new ArrayList<>();
    private ArrayList<List<Double>> dataSet = new ArrayList<>();

    public kNN() {}

    private double getEuclideanDistance(List<Double> src, List<Double> target)
    {
        if(src.size() != target.size())
        {
            System.out.println("长度不一致, 无法计算欧氏距离!");
            return -1;
        }
        int size = src.size();
        double result = 0.0;
        for(int i=0; i<size; i++)
        {
            double temp = src.get(i) - target.get(i);
            result += Math.pow(temp, 2);
        }
        result = Math.sqrt(result);
        return result;
    }

    private void loadDataSet()
    {
        String encoding = "utf-8";
        File f = new File("E:\\Program\\IDEA\\Algorithm\\kNN\\src\\" +
                "cn\\edu\\uestc\\algorithm\\knn\\trainset.txt");
        if(f.isFile() && f.exists())
        {
            try {
                InputStreamReader reader = new InputStreamReader(new FileInputStream(f), encoding);
                BufferedReader bufferedReader = new BufferedReader(reader);
                String textLine;
                while ((textLine = bufferedReader.readLine()) != null)
                {
                    String[] arrays = textLine.trim().split(",| ");
                    List<String> temp = Arrays.asList(arrays);
                    List<Double> data = new ArrayList<>();
                    for(String i: temp)
                    {
                        if(temp.indexOf(i) == (temp.size() - 1))
                            break;
                        data.add(Double.valueOf(i));
                    }
                    labels.add(temp.get(temp.size() - 1));
                    dataSet.add(data);
                }
                bufferedReader.close();
                reader.close();
                System.out.println(dataSet);
                System.out.println(labels);
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
        else
        {
            System.out.println("File Not Found!");
        }
    }

    private double getNormalized_0_1_Data(double src, double min, double max)
    {
        return ((src - min)/(max - min));
    }

    private void normalizeDataSet(List<Double> inX)     // 将dataSet和输入向量均归一化数值
    {
        loadDataSet();
        int dataSetSize = dataSet.size();
        if(dataSetSize == 0)
            throw new IndexOutOfBoundsException("The Size of DataSet is 0 !");
        int itemSize = dataSet.get(0).size();
        for(int i=0; i<itemSize; i++) {
            List<Double> temp = new ArrayList<>();
            for (List<Double> aDataSet : dataSet) {
                temp.add(aDataSet.get(i));
            }
            temp.add(inX.get(i));
            List<Double> temp_copy = new ArrayList<>();
            temp_copy.addAll(temp);
            Collections.sort(temp_copy, Double::compareTo);
            int tempSize = temp.size();
            double max = temp_copy.get(tempSize - 1);
            double min = temp_copy.get(0);
            for(int n=0; n<tempSize; n++)
            {
                temp.set(n, getNormalized_0_1_Data(temp.get(n), min, max));
            }
            int m;
            for(m=0; m<dataSetSize; m++)
            {
                dataSet.get(m).set(i, temp.get(m));
            }
            inX.set(i, temp.get(m));
        }
        System.out.println(dataSet);
        System.out.println(inX);
    }

    private double getWeight(double distance)
    {
        return 1 / (Math.pow(distance, 2));
    }

    public String kNNClassifier(List<Double> inX, int k)
    {
        normalizeDataSet(inX);
        Map<Integer, Double> distanceMap = new HashMap<>();
        for(List<Double> item: dataSet)
        {
            double distance = getEuclideanDistance(inX, item);
            distanceMap.put(dataSet.indexOf(item), distance);
        }
        List<Map.Entry<Integer, Double>> entryList = new ArrayList<>(distanceMap.entrySet());
        Collections.sort(entryList, (o1, o2) -> o1.getValue().compareTo(o2.getValue()));
        System.out.println(entryList);

        Map<String, Double> labelCount = new HashMap<>();
        for(int i=0; i<k; i++)
        {
            String label = labels.get(entryList.get(i).getKey());
            /**
             * 根据距离的远近, 对近邻的投票票数进行加权, 距离越近则权重越大
             * @see #getWeight(double) 权重为距离平方的倒数
             **/
            labelCount.put(label,
                    (labelCount.getOrDefault(label, 0.0) + getWeight(entryList.get(i).getValue()))
            );
        }
        List<Map.Entry<String, Double>> entries = new ArrayList<>(labelCount.entrySet());
        Collections.sort(entries, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        System.out.println(entries);
        return entries.get(0).getKey();
    }
}
