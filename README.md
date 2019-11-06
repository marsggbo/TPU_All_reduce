# TPU_All_reduce
测量TPU Allreduce带宽

# 文件说明

- mnist_tpu.py: 用于测试的主文件
- parse_gzfile.py: 用于解析生成的log文件，得到通信带宽


- p*.sh：用于跟踪测量对应TPU的计算流程，例如p8.sh要配合run8.sh使用，具体方法见下面的说明。
- run*.sh: 控制运行mnist_tpu.py的脚本文件，run8.sh表示用TPUv2-8运行，其他同理。在运行时只需要指定一个参数，即scale。该参数用于控制模型（即通信数据）的大小的变化，即`Model size=0.5*scale`。

```
./run8.sh 1
```
表示用TPUv2-8运行，模型大小是0.5MB。

- parsedLog.log: 整理得到的各个TPU的带宽计算结果。

# 实验方法

建议使用tmux创建两个独立的窗口，

在一个窗口下运行：`./run8.sh 1`

在另一个窗口下运行：`./p8.sh 1`

将参数分别从1调整为 2,4,8,16,32,64,128,256,512,1024


每次运行p*.sh文件都会在对应指定的路径下生成一个`plugins`的文件夹，其结构如下：

![](https://ask.qcloudimg.com/draft/1215004/e697hmzjhk.png)

```bash
plugins
|___profile
    |___2019.10.14.15.56.32（以时间命名的文件夹）
        |__*overview_page.json
        |__*input_pipeline.json
        |__*op_profile.json
        |__*.trace.json.gz
        |__*.trace
        |__*.tracetable
```

parse_gzfile.py 文件就是对`trace.json.gz`文件进行解析并得到最终的带宽，其实现原理如下：

- `cp()`函数: 将各个目录下的`*.trace.json.gz`文件拷贝到当前目录,例如`mnist_tpuv2n8_bs1024_scale-256/10.0.0.102.trace.json.gz`
- 拷贝完数据后对`gz`文件使用`parse()`函数进行遍历解析， 最终得到一个json格式的变量
- 最后将上面的变量保存至`log.json`文件中

