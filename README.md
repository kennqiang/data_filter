# data_filter 项目
This is a repository providing data filtering methods by one-pass clustering algorithm. TF-IDF and w2v features are used.

## 如何运行？
将整个项目拷贝到自己的目录中，在该目录中建一个data文件夹，data文件夹中放入需要处理的数据文件

## pay attention!
代码针对的数据文件格式是json格式
例如：
[
{'id':0, 'content': 'this is the first sentence'},
{'id':1, 'content': 'this is the second sentence'},
......
{'id':99, 'content': 'this is the last sentence'}
]

如果文件格式有变化，需要在get_tfidf.py和data_filter.py中做相应的修改
