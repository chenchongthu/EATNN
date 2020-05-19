# EATNN

This is our implementation of the paper:

*Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Yiqun Liu and Shaoping Ma. 2019. [An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation.](http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf) 
In SIGIR'19.*

**Please cite our SIGIR'19 paper if you use our codes. Thanks!**

```
@inproceedings{chen2019efficient,
  title={An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation},
  author={Chen, Chong and Zhang, Min and Wang, Chenyang and Ma, Weizhi and Li, Minming and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={225--234},
  year={2019},
  organization={ACM}
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python
- Tensorflow
- numpy
- pandas


## Example to run the codes		

Train and evaluate the model:

```
python EATNN.py
```

## Suggestions for parameters

The followling important parameters need to be tuned for different datasets, which are:

```
self.weight1=0.1
self.weight2=0.1
self.mu=0.1
deep.dropout_keep_prob
```

Specifically, we suggest to tune "self.weight" among \[0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]. It's also acceptable to simply make the two weights the same. Generally, this parameter is related to the sparsity of dataset. If the dataset is more sparse, then a small value of negative_weight may lead to a better performance.

The coefficient parameter self.mu determines the importance of different tasks in joint learning. It can be tuned among \[0.1,0.3,0.5,0.7,0.9].

Generally, the performance of our EATNN is very good. You can also contact us if you can not tune the parameters properly.




Last Update Date: May 19, 2020
