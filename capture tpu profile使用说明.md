TPU Profile Tool 介绍


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

# 性能剖析

性能剖析 (Profile) 标签页在您运行 capture_tpu_profile 时创建，此标签页仅在您捕获某些模型数据后才会显示在 TensorBoard 中。数据可用后，点击性能剖析 (Profile) 标签页会显示一系列有助于性能分析的工具：

概览页面
Input Pipeline Analyzer
XLA Op Profile
Trace Viewer（仅限 Chrome 浏览器）
Memory Viewer
Pod Viewer
Streaming Trace Viewer（仅限 Chrome 浏览器）

## 性能剖析概览页面
概览页面 (overview_page) 位于性能剖析 (Profile) 下方，其提供模型在捕获运行期间的执行情况汇总。该页面展示所有 TPU 的聚合概览页，以及整体输入流水线分析。您可以在主机下拉列表中选择单个 TPU。

该页面在以下面板中显示数据：

![image.png](https://ask.qcloudimg.com/draft/1215004/6yv8n0n0pk.png)

- **性能摘要 (Performance summary)**
    - 所有采样步的平均单步用时
    - 主机空闲时间百分比
    - TPU 空闲时间百分比
    - TPU 矩阵单元利用率百分比
- **单步用时图** (Step-time graph)。展示所有采样步的设备单步用时（以毫秒为单位）图。蓝色区域对应的是 TPU 空闲时等待主机输入数据的单步用时部分。橙色区域展示 Cloud TPU 实际运行的时间。

- **TPU 上最耗时的 10 个 TensorFlow 操作** (Top 10 TensorFlow operations on TPU)。展示占用了大部分时间的 TensorFlow 操作。点击显示表格 (Show table) 按钮将显示如下表格：

![image.png](https://ask.qcloudimg.com/draft/1215004/fobbqslkqy.png)

每行显示操作的自用时间（以所有操作所用时间的百分比的形式）、累计时间、类别、名称以及实现的 FLOPS 率。

- **运行环境 (Run environment)**

    - 使用的主机数
    - 使用的 TPU 类型
    - TPU 核心数量
    - 训练批次大小
- **后续步骤建议** (Recommendation for next steps)。报告模型输入是否受限，以及是否出现 Cloud TPU 问题。推荐可用于找出性能瓶颈的工具。

## Input Pipeline Analyzer

Input Pipeline Analyzer 可提供有关性能结果的数据分析。该工具展示 capture_tpu_profile 工具收集的 `input_pipeline.json` 文件的性能结果。

该工具能够立即告知您程序是否输入受限，并可指导您完成设备端和主机端分析，以调试并确定流水线的哪个阶段存在瓶颈。

如需深入了解如何优化流水线性能，请参阅有关[输入流水线性能](https://www.tensorflow.org/versions/master/performance/datasets_performance)的指南。

输入流水线
当 TensorFlow 程序从文件中读取数据时，数据以流水线的方式从 TensorFlow 图的顶部开始读取。读取过程分为多个串联连接的数据处理阶段，其中一个阶段的输出是下一个阶段的输入。这个读取系统称为**输入流水线**。

从文件中读取记录的典型流水线具有以下阶段：

    - 文件读取
    - 文件预处理（可选）
    - 将文件从主机传输到设备
效率低下的输入流水线可能会严重降低应用的速度。当应用在输入流水线上花费大量时间时，我们将其称为**输入受限**。使用 Input Pipeline Analyzer 可以确定输入流水线的低效位置。

**输入流水线信息中心**

如需打开 Input Pipeline Analyzer，请选择性能剖析 (Profile)，然后从工具 (Tools) 下拉列表中选择 input_pipeline_analyzer。

信息中心包含三个版块：

![image.png](https://ask.qcloudimg.com/draft/1215004/la0d2259e7.png)

1.摘要。 展示整个输入流水线的摘要信息，其中包含您的应用是否输入受限以及受限程度（如果受限）的信息。

2.设备端分析。 展示详细的设备端分析结果，包括设备单步用时以及每一步中等待所有核心的输入数据所占用的设备时间的范围。

3.主机端分析。 展示主机端的详细分析，包括主机上输入处理时间的明细。

**输入流水线摘要**

第 1 个版块显示等待主机输入所占用的设备时间百分比，以报告您的程序是否输入受限。如果您使用的是已经过测试的标准输入流水线，该工具会报告大部分输入处理时间用于何处。例如：

![image.png](https://ask.qcloudimg.com/draft/1215004/n85z5mlwia.png)

**设备端分析**

第 2 个版块显示设备端分析的详细信息，让您深入了解在设备和主机上花费的时间，以及等待主机输入数据所占用的设备时间。

![image.png](https://ask.qcloudimg.com/draft/1215004/ta3gysx02e.png)

1.根据步编号绘制的单步用时。展示所有采样步的设备单步用时（以毫秒为单位）图。蓝色区域对应的是 Cloud TPU 空闲时等待主机输入数据的单步用时部分。橙色区域展示 Cloud TPU 实际运行的时间。

2.单步用时统计信息。报告设备单步用时的平均值、标准偏差和范围（[最小值，最大值]）。

3.根据步编号绘制的等待所有核心的输入数据占用的设备时间。展示等待输入数据处理所花费的设备时间（表示为总设备单步用时的百分比）的折线图。由于不同核心所花费时间的比例不尽相同，因此系统会针对每一步绘制不同核心的比例范围。由于单步用时由最慢的核心决定，因此这个范围应尽可能小。

4.等待输入数据的时间比例。报告设备等待输入数据所用的时间比例的平均值、标准偏差和范围（[最小值，最大值]），基于设备单步用时总时长进行归一化。


**主机端分析**

第 3 个版块显示主机端分析的详细信息，它将主机上的输入处理时间（Dataset API 操作所用时间）细分为几类：

- 按需从文件中读取数据。在没有缓存、预取和交错的情况下从文件读取数据所用的时间。
- 预先从文件中读取数据。读取文件所花费的时间，包括缓存、预取和交错。
- 数据预处理。用于预处理操作的时间，例如图片解压缩。
- 将要传输到设备的数据加入队列。先将数据加入馈入队列然后再向设备传输数据所花费的时间。


![image.png](https://ask.qcloudimg.com/draft/1215004/shsizrqgpz.png)

如需按执行时间明细查看各个输入操作及其类别的统计信息，请点击“显示输入操作统计信息”(Show Input Op Statistics) 按钮。

系统将显示类似如下的源数据表：


![image.png](https://ask.qcloudimg.com/draft/1215004/mavagz8hcw.png)

每个表条目包含以下信息：

1. 输入操作。显示输入操作的 TensorFlow 操作名称。
2. 计数。显示在性能剖析期间执行的操作的实例总数。
3. 总时间（以毫秒为单位）。显示每个实例所用的时间的累计总和。
4. 总时间百分比。表示在一个操作上花费的总时间占输入处理总时间的比例。
5. 总自用时间（以毫秒为单位）。显示每个实例所用的自用时间的累计总和。这里的自用时间测量在函数体内部所用的时间，不包括它调用的函数所用的时间。例如，Iterator::PaddedBatch::Filter::ForeverRepeat::Map 由 Iterator::PaddedBatch::Filter 调用，因此前者的总自用时间不包括在后者的总自用时间内。
6. 总自用时间百分比。表示总自用时间占输入处理总时间的比例。
7. 类别。显示输入操作的处理类别。


## Op Profile

Op Profile (op_profile) 是一款 Cloud TPU 工具，此工具可以展示在性能剖析期间执行的 XLA 操作的性能统计信息。Op Profile 可展示以下内容：

您的应用使用 Cloud TPU 的程度，按类别和 TPU FLOPS 利用率显示各个操作所花费的时间的百分比。
最耗时的操作。这些操作可能有待优化。
各个操作的详细信息，包括形状、填充和使用该操作的表达式。
您可以借助 Op Profile 找到可能需要进行优化的目标。例如，如果您的模型的 TPU 峰值 FLOPS 仅达到 5%，您可以使用此工具找出执行时间最长的 XLA 操作并查看它们使用了多少 TPU FLOPS。

**使用 Op Profile**

在收集性能剖析文件时，capture_tpu_profile 还会收集包含 XLA 操作的性能统计信息的 op_profile.json 文件。

如需在 TensorBoard 中查看 op_profile 中的数据，请点击屏幕顶部的性能剖析 (Profile) 标签页，然后从工具 (Tools) 下拉列表中选择 op_profile。您将看到如下所示的界面：


![image.png](https://ask.qcloudimg.com/draft/1215004/6dzvhub0pa.png)

1. 概览部分。显示 Cloud TPU 计算潜力的使用百分比，并提供优化建议。
2. 控制面板。包含一个用于控制操作表中显示的操作数的设置滑块，以及一个用于设置操作表，使其仅列出总执行时长前 90% 以内的操作的切换开关。
3. 操作表。按总时长列出与 XLA 操作相关联的顶级 TensorFlow 操作类别，表示为 Cloud TPU 使用率百分比（前提是该类别中的所有操作都要执行）。
4. 操作详细信息卡片。将鼠标悬停在表中的某个操作上时，会出现一张卡片，上面显示有关此操作的详细信息，包括 FLOPS 利用率、使用该操作的表达式以及操作布局（匹配）。

 
**XLA 操作表**

操作表按照 Cloud TPU 使用率百分比从高到低的顺序列出 XLA 操作类别。对于每个类别，该表都会默认显示花费时间的百分比、操作类别名称、关联的 TensorFlow 操作的名称及其 FLOPS 利用率百分比。如需显示（或隐藏）某个类别中 10 个最耗时的 XLA 操作，请点击表格中类别名称旁边的三角形。


![image.png](https://ask.qcloudimg.com/draft/1215004/jx62wbwf0x.png)

1. 时间。显示该类别中所有操作所用时间的总百分比。您可以点击以展开条目，查看每个操作所用时间的明细。
2. 水平栏。显示各个类别的时间分布。
3. 10 个最耗时的操作。点击类别名称旁边的切换开关可以显示或隐藏此类别中 10 个最耗时的操作。如果操作列表中显示了融合操作条目，您可以将其展开以查看其包含的非融合元素级操作。
4. TensorFlow 操作。显示与 XLA 操作关联的 TensorFlow 操作名称。
5. FLOPS。显示 FLOPS 利用率，即以 Cloud TPU 峰值 FLOPS 的百分比表示的 FLOPS 的测量值。FLOPS 利用率百分比越高，操作运行速度越快。表单元格采用颜色编码：绿色表示 FLOPS 利用率高（好），红色表示 FLOPS 利用率低（差）。


**操作详细信息卡片**

将鼠标悬停在表条目上时，左侧会显示一张卡片，上面显示有关 XLA 操作或操作类别的详细信息。典型的卡片如下所示：


![image.png](https://ask.qcloudimg.com/draft/1215004/jvw3sjb9bd.png)

1. 名称。突出显示 XLA 操作的名称。
2. 类别。显示该操作的类别。
3. FLOPS 利用率。显示 FLOPS 利用率占总 FLOPS 的百分比。
4. 表达式。显示包含该操作的 XLA 表达式。
5. 内存利用率。显示程序使用的峰值内存占总内存的百分比。
6. 布局（仅限卷积运算。）显示张量的形状和布局，包括张量的形状是否与矩阵单元完全匹配以及矩阵的填充方式。

**解读结果**

对于卷积运算，TPU FLOPS 利用率低可能是由以下原因之一或全部造成的：

- 填充（矩阵单元仅被部分使用）
- 卷积运算受限于内存


本部分简要解释另一个 FLOPS 较低的模型中的某些数字。在这个例子中，输出融合和卷积占用了大部分执行时间，同时还有一个 FLOPS 非常低的矢量或标量操作的长尾。

此类性能剖析文件的一种优化策略是将矢量或标量操作转换为卷积运算。

在以下示例中，%convolution.399 显示的 FLOPS 利用率和内存利用率低于上例中的 %convolution.340。


![image.png](https://ask.qcloudimg.com/draft/1215004/oemioa2db1.png)

仔细观察布局可发现，批次大小 16 被填充至 128，并且特征大小 3 被填充至 8，这表明只有 5% 的矩阵单元被有效利用。（此例中利用率百分比的计算方法是将批次大小乘以特征大小，即 16 乘 3，然后用结果除以填充值，即先除以 128，再除以 8。）将此示例中的 FLOPS 与前例中的 %convolution.340 进行比较，后者与矩阵完全匹配。

**Pod Viewer**

Pod Viewer 工具为 Pod 中的每个核心提供性能可视化，并显示 Pod 中核心之间的通信通道的状态。Pod Viewer 可以识别并突出显示潜在的瓶颈和需要优化的区域。此工具适用于完整 Pod 和所有 v2 和 v3 Pod 切片。

如需显示 Pod Viewer 工具，请执行以下操作：

1. 从 Tensorboard 窗口右上角的菜单按钮中选择性能剖析 (Profile)。


![image.png](https://ask.qcloudimg.com/draft/1215004/by0hnajwqe.png)

2. 点击窗口左侧的工具 (Tools) 菜单，然后选择 pod_viewer。


![image.png](https://ask.qcloudimg.com/draft/1215004/tiwx9v4q22.png)

**Pod Viewer** 用户界面包括以下内容：

1. 步滑块，用于选择要检查的步。
2. 拓扑图，以交互方式直观显示整个 TPU 系统中的 TPU 核心。
3. 通信链接图表，用于直观显示拓扑图中的发送和接收 (recv) 通道。
4. 发送和接收通道延迟时间的条形图。将鼠标悬停在此图表中的某个柱形上会激活通信链接图表中的通信链接。左侧栏上会显示一个通道详细信息卡片，上面显示该通道的详细信息，例如传输的数据的大小、延迟时间和带宽。
5. 步明细图表，直观显示所有核心的步明细信息。此图表可用于跟踪系统瓶颈以及确定特定核心是否降低系统速度。

**步滑块**


![image.png](https://ask.qcloudimg.com/draft/1215004/jwflf29ji7.png)

使用滑块选择一个步。该工具的其余部分显示该步的统计信息，例如步明细和通信链接。

拓扑图
拓扑图由主机、芯片和核心以分层方式进行组织。最小的矩形是 TPU 核心。合并在一起的两个核心表示一个 TPU 芯片，四个芯片表示一个主机。


![image.png](https://ask.qcloudimg.com/draft/1215004/ej7rkp4ciz.png)

拓扑图也是热图，其颜色按特定明细（例如高 FLOPS 计算、馈入、发送等）在所选步中占用时间的百分比进行标示。拓扑图下方的柱形（如下图所示）展示了核心和芯片使用率的颜色标示方式。核心的颜色从黄色向蓝色过渡，以显示不同利用率。对于高 FLOPS 计算，数字越大（颜色越暗）表示计算需要花费的时间越长。对于所有其他明细，数字越小（颜色越浅）表示等待时间越短。当某个核心比其他核心颜色更暗时，说明其为潜在问题区域或热点。

点击系统名称旁边的下拉菜单选择器（图中已圈出），选择要检查的明细的特定类型。

将鼠标悬停在任何小矩形（单个核心）上，会出现一个信息提示框，其上显示核心在系统中的位置、其全局芯片 ID 和主机名。信息提示框中的内容还包括所选明细类别（例如高 FLOPS）的时长，及其在步中的利用率百分比。


![image.png](https://ask.qcloudimg.com/draft/1215004/up6kuvrzjg.png)

通信通道

如果您的模型使用发送和接收链接在核心之间进行通信，则此工具有助于直观呈现这些链接。当您的模型包含发送和接收操作时，您可以使用通道 ID 选择器来选择通道 ID。来自源 (src) 核心和目标 (dst) 核心的链接表示通信通道。将鼠标悬停在图表的柱形上，可以在拓扑图上呈现通信通道，以显示发送和接收通道的延迟时间。


![image.png](https://ask.qcloudimg.com/draft/1215004/sn8k9mnjpy.png)

左侧栏会出现一张卡片，上面显示有关通信通道的详细信息。典型的卡片如下所示：


![image.png](https://ask.qcloudimg.com/draft/1215004/5lg2x3bmvj.png)

1. 已传输的数据 (Data Transferred)，显示发送和接收通道已传输的数据量，以兆比字节 (MiB) 为单位。
2. 延迟时间 (Latency)，显示从发送事件开始到接收完成事件结束的时长，以微秒为单位。
BW，显示在运行时长内从源核心到目标核心传输的数据量，以吉比字节 (GiB) 为单位。
3. 发送延迟 (Send Delay)，指从接收完成开始到发送开始的时长，以微秒为单位。如果接收完成操作在发送操作开始之后开始，则延迟为零。
4. HLO 名称 (Hlo Names)，显示与此通道关联的 XLA HLO 操作名称。这些 HLO 名称与其他 TensorBoard 工具（如 op_profile 和 memory_viewer）中显示的统计信息相关联。


发送和接收通道延迟时间图表

此图表提供有关发送和接收通信通道的详细信息。将鼠标悬停在此图表上的某个柱形上可显示上述拓扑图上的发送和接收链接。


![image.png](https://ask.qcloudimg.com/draft/1215004/j1u44rog7h.png)

步明细图表
此图表提供了每个训练或评估步的详细信息。

x 轴表示全局芯片 ID，y 轴表示时间（以微秒为单位）。在此图表中，您可以看到在特定训练步中时间用于何处、何处存在瓶颈，以及所有芯片之间是否存在负载不平衡。


![image.png](https://ask.qcloudimg.com/draft/1215004/i5z06flxxa.png)

左侧栏会出现一张卡片，上面显示有关步明细的详细信息。典型的卡片如下所示：


![image.png](https://ask.qcloudimg.com/draft/1215004/kkxyv2o5pr.png)

卡片中的字段展示以下内容：

- 高 FLOPS 计算 (High Flops Compute)，即卷积运算或输出融合操作 (ops) 所花费的时间。
- 低 FLOPS 计算 (Low flops compute)，通过从总时长中减去所有其他明细来计算。
- 馈入 (Infeed)，即 TPU 在主机上等待的时间。
- 馈出 (Outfeed)，即主机等待 TPU 输出的时间。
- AllReduce 同步 (AllReduce sync)，这是等待与其他核心同步的 CrossReplicaSum 操作所花费的时间。CrossReplicaSum 操作计算各个副本的总和。
- AllReduce 计算 (AllReduce compute)，这是 CrossReplicaSum 操作实际花费的计算时间。
- 芯片到芯片发送操作 (Chip to chip send ops)，这是发送操作所花费的时间。
- 芯片到芯片接收完成操作 (Chip to chip recv-done ops)，这是接收完成操作所花费的时间。


## Trace viewer

Trace viewer 是性能剖析 (Profile) 下的 Cloud TPU 性能分析工具。此工具使用 Chrome 跟踪记录事件分析查看器，因此仅适用于 Chrome 浏览器。

Trace Viewer 包含一个时间轴，其中显示的内容包括：

- 由 TensorFlow 模型执行的操作的时长。
- 系统的哪个部分（TPU 还是主机）执行了操作。通常，主机执行馈入操作，它会预处理训练数据并将其传输到 TPU，而 TPU 执行实际的模型训练。
您可以通过 Trace Viewer 找出模型中的性能问题，然后采取措施来加以解决。例如，从总括层面来讲，您可以确定占用大部分时间的是馈入还是模型训练。展开细目，您可以确定哪些 TensorFlow 操作执行时间最长。

请注意，Trace Viewer 针对每个 Cloud TPU 仅限访问 100 万个事件。如果您需要访问更多事件，请使用 Streaming Trace Viewer。

**Trace Viewer 界面**

如需打开 Trace Viewer，请转到 TensorBoard 并点击屏幕顶部的性能剖析 (Profile) 标签页。Trace Viewer 默认显示最近的运行：


![image.png](https://ask.qcloudimg.com/draft/1215004/by0hnajwqe.png)

此屏幕包含以下主要元素（上图中标有数字的部分）：

- 运行下拉列表。包含已捕获其跟踪记录信息的所有运行。默认视图为最近的运行，您也可以打开下拉列表选择其他运行。
- 工具下拉列表。选择不同的性能剖析工具。
- 主机下拉列表。选择包含 Cloud TPU 集的主机。
- 时间轴窗格。显示 Cloud TPU 和主机随时间执行操作的情况。
- 详细信息窗格。显示时间轴窗格中所选操作的更多信息。


以下是时间轴窗格的详细信息：


![image.png](https://ask.qcloudimg.com/draft/1215004/7j97u60e8n.png)

时间轴窗格包含以下元素：

- 顶栏。包含各种辅助控件。
- 时间轴。显示相对于跟踪记录开始时的时间。
- 版块和跟踪记录标签。每个版块包含多个跟踪记录，左侧有一个三角形，点击此三角形可展开和收起该版块。系统中的每- 个处理元素都有一个版块。
- 工具选择器。包含与 Trace Viewer 交互的各种工具。
- 事件。这些事件显示操作的执行时间或元事件（例如训练步）的时长。
- 垂直标签栏。它对 Cloud TPU 没有用处。此栏是 Chrome 提供的通用跟踪记录查看器工具的一部分，用于各种性能分析任务。


**版块和跟踪记录**

Trace Viewer 包含以下版块：

- 每个 TPU 节点有一个版块，其标签为 TPU 芯片的编号及芯片内的 TPU 节点的编号（例如“Chip 2: TPU Core 1”）。- 每个 TPU 节点版块包含以下跟踪记录：
  - 步。显示 TPU 上运行的训练步的时长。
  - TensorFlow 操作。显示在 TPU 上执行的 TensorFlow 操作。
  - XLA 操作。显示在 TPU 上运行的 XLA 操作。（每个 TensorFlow 操作都会转换为一个或多个 XLA 操作。XLA 编译器- 会将这些 XLA 操作转换为在 TPU 上运行的代码。）
- 针对在主机的 CPU 上运行的线程的版块，其标签为“Host Threads”。对于每个 CPU 线程，此版块都包含一个跟踪记录。注意：您可以忽略版块标签旁边显示的信息。

## 监控 Cloud TPU 作业

本部分介绍如何使用 capture_tpu_profile 捕获单个性能剖析文件或在命令行界面上实时持续监控 Cloud TPU 作业。将 --monitoring_level 选项设置为 0（默认值）、1 或 2，可以分别获得单个性能剖析文件、基本监控数据或详细监控数据。

注意：您无法同时捕获性能剖析文件并监控作业。
- 打开一个新的 Cloud Shell 并使用 ssh 连接到您的虚拟机（将命令中的 $vm 替换为您的虚拟机名称）：

`gcloud compute ssh $vm --ssh-flag=-L6006:localhost:6006`

- 在新的 Cloud Shell 中，运行 capture_tpu_profile 并将 --monitoring_level 标志设置为 1 或 2，例如：

`(vm)$ capture_tpu_profile --tpu=$TPU_NAME  --monitoring_level=1`

设置 monitoring_level=1 会生成类似于以下内容的输出：

    TPU type: TPU v2
    Utilization of TPU Matrix Units is (higher is better): 10.7%
设置 monitoring_level=2 会显示更详细的信息：

    TPU type: TPU v2
    Number of TPU Cores: 8
    TPU idle time (lower is better): 0.091%
    Utilization of TPU Matrix Units is (higher is better): 10.7%
    Step time: 1.95 kms (avg), 1.90kms (minute), 2.00 kms (max)
    Infeed percentage: 87.5% (avg). 87.2% (min), 87.8 (max)

监控标志

- --tpu（必需）指定要监控的 Cloud TPU 的名称。
- --monitoring_level。将 capture_tpu_profile 的行为从生成单个性能剖析文件更改为生成基本或详细的持续监控数- 据。此标志有以下三个等级： 第 0 级（默认）：生成一个性能剖析文件，然后退出。 第 1 级：显示 TPU 版本和 - TPU 利用率。 第 2 级：显示 TPU 利用率、TPU 空闲时间和使用的 TPU 核心数量。同时提供单步用时的最小值、平均- 值和最大值，以及馈入百分比贡献。
- --duration_ms（可选；默认值为 1000ms）指定在每个周期内分析 TPU 主机的时间。通常，这个时长应足够长，以便捕- 获至少一个训练步的数据。在大多数模型中，1 秒可捕获一个训练步，但如果模型单步用时非常大，则可以将此值设置为 - step_time 的 2 倍（以毫秒为单位）。
- --num_queries 指定运行 capture_tpu_profile 周期数。如需持续监控 TPU 作业，请将此值设置为较大的数字。如需快速检查模型的单步用时，请将此值设置为较小的数字。