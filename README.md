# FMFCC Audio Deepfake Detection Solution
This repo provides an solution for the [多媒体伪造取证大赛](http://fmfcc.net). 
Our solution achieve the 1st in the Audio Deepfake Detection track .
The ranking can be seen [here](http://fmfcc.net/contest-introduction)

## Authors
Institution: Shenzhen Key Laboratory of Media Information Content Security(MICS)

Team name: Forensics_SZU  

Username: 
- [Baoying Chen](https://github.com/beibuwandeluori) email: 1900271059@email.szu.edu.cn
- [Yuankun Huang]
- [Siyu Mei]
## Pipeline
### EfficientNet-B2 + mel features (不同mel参数的训练两个模型ensemble)
![image](pipeline.png)
1) 对音频进行分割，每隔 0.96 秒提取音频 mel 特征（维度为[1, 96, 96]）；
2) 使用EfficientNet-B2作为分类模型，修改输入通道为1，加载ImageNet的预训练权值；
3) 分类模型采取两种方式训练， Model1 是窗口长度和步长都是 0.96s，无重叠分割音频，Mode2 是窗口为 0.96s,而步长为 0.48 秒，有重叠分割音频；
4) 一个完整音频的mel特征维度为[n, 96, 96], 每个模型对n个[1, 96, 96]特征预测后求均值得到预测值，然后将两种模型对音频的预测值的平均值作为最终的预测值。
