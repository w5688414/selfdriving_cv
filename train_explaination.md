
# 主代码 train.py 函数
    该代码用于训练imitation learning的网络，网络的输入为图像数据，speed和command，网络输出为steer gas brake。
## 参数：
- trainScratch = True # params[0]--True表示不重载训练参数；False表示重载训练参数 
- dropoutVec  # params[1] 输入矢量23维
- image_cut=[115, 510] # params[2]
- learningRate = 0.0002 # params[3] multiplied by 0.5 every 50000 mini batch
- beta1 = 0.7 # params[4]
- beta2 = 0.85 # params[5]
- num_images  # params[6] 图像数目:200*3289=657800
- iterNum = 294000 # params[7]
- batchSize = 120 -# params[8] size of batch 
- valBatchSize = 120 # params[9] size of batch for validation set
- NseqVal = 5  # params[10] number of sequences to use for validation
- epochs = 100 # params[11]
- samplesPerEpoch = 500 # params[12]
- L2NormConst = 0.001 # params[13]

params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2, num_images, iterNum, batchSize, valBatchSize, NseqVal, epochs, samplesPerEpoch, L2NormConst]


- timeNumberFrames = 1 --- number of frames in each samples
- prefSize = _image_size = (88, 200, 3) # 图像的大小88*200*3
- memory_fraction=0.25 # 和内存有关的参数
- cBranchesOutList = ['Follow Lane','Go Left','Go Right','Go Straight','Speed Prediction Branch'] #
- controlInputs = [2,5,3,4] # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
- branchConfig = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]


## 配置GPU
config = tf.ConfigProto(allow_soft_placement=True)
## 准备数据
 - datasetDirTrain = '/home/ydx/AutoCarlaData/AgentHuman/SeqTrain/' # 训练集目录
 - datasetDirVal = '/home/ydx/AutoCarlaData/AgentHuman/SeqVal/' # 测试集目录
 - datasetFilesTrain = glob.glob(datasetDirTrain+'*ddd.h5') # 训练集文件总共3289
 - datasetFilesVal = glob.glob(datasetDirVal+'*.h5') # 测试集文件总共374
### 网络输入数据
 - inputs： 维度为2的列表
     - input[0]=input_images 一维表示输入的图像inputImages 88*200*3；
     - input[1]=input_data 
         input_data[0]: 一维表示输入的控制量input_control (维度为4的one-hot矢量)   controlInputs = [2,5,3,4] # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight), 比如[0 1 0 0]表示 Straighut
         input_data[1]: 测量量input_speed（一个维度）
### 网络输出数据/网络标签值
 - targets：维度为2的列表
         targets[0]: 一维表示target_speed(维度为1)
         targets[1]: 一维表示target_control（维度为3）

## 创建网络
- netTensors = Net(branchConfig, params, timeNumberFrames, prefSize)
  - netTensors = {
     - 'inputs': inputs, 
     - 'targets': targets,
     - 'params': params,
     - 'dropoutVec': dout, 
     - 'output': controlOpTensors
       }
- controlOpTensors= {
     - 'optimizers': contSolver, 
     - 'losses': contLoss, 总的损失值
     - 'output': networkTensor, 网络的输出值
     - 'print': pr % 应该表示的是命令
       }
## 训练网络
- epoch: 运行次数
- steps: 运行的第几个batch
### 选择数据
- cur_branch： 根据Branch的数目产生一个随机数（若有4个Branch则该随机数大于等于0小于等于3）确定想要训练哪一个Branch
    - cur_branch = np.random.choice(branch_indices)
- 根据随机产生的cur_branch数值提取训练数据：
    - xs, ys = next(genBranch(fileNames=datasetFilesTrain,branchNum=controlInputs[cur_branch],batchSize=batchSize))
### 增强数据
×××  参考https://blog.csdn.net/u012897374/article/details/80142744×××

- st = lambda aug: iaa.Sometimes(0.4, aug) # iaa.Sometimes 对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters。
- oc = lambda aug: iaa.Sometimes(0.3, aug)
- rl = lambda aug: iaa.Sometimes(0.09, aug)
- seq = iaa.Sequential(
    - [rl(iaa.GaussianBlur((0, 1.5))), # 高斯 blur images with a sigma between 0 and 1.5
    - rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
    - oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
    - oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
    - oc(iaa.Add((-40, 40), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
    - st(iaa.Multiply((0.10, 2.5), per_channel=0.2)), # change brightness of images (X-Y% of original value)
    - rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast], 
    - random_order=True)
- xs = seq.augment_images(xs)
### 输入数据
- inputData.append(sess.run(tf.one_hot(ys[:, 24], 4)))  # Command Control, 4 commands
- inputData.append(ys[:, 10].reshape([batchSize, 1]))  # Speed
- feedDict = {
    - netTensors['inputs'][0]: xs, 
    - netTensors['inputs'][1][0]: inputData[0],
    - netTensors['inputs'][1][1]: inputData[1], 
    - netTensors['dropoutVec']: dropoutVec,
    - netTensors['targets'][0]: ys[:, 10].reshape([batchSize, 1]),
    - netTensors['targets'][1]: ys[:, 0:3]}
### 训练
_, p, loss_value = sess.run([contSolver, pr, contLoss], feed_dict=feedDict)
### 策略
 - 每10个Batch输出训练结果
 - 每500个Batch输出验证结果
 - 每250个Batch保存Checkpoint结果
 - 每50000个Batch学习速率减半

# 函数---Net
## 输入
 - branchConfig # 4个Branch的
 - params # 参数
 - timeNumberFrames 
 - prefSize # 图像大小
## 输出
 tensors = {
  - 'inputs': inputs, 
  - 'targets': targets,
  - 'params': params,
  - 'dropoutVec': dout,
  - 'output': controlOpTensors}
## 重要中间变量
 - inputs： 维度为2的列表
     - input[0]=input_images 一维表示输入的图像inputImages 88*200*3；
     - input[1]=input_data 
        - input_data[0]: 一维表示输入的控制量input_control (维度为4的one-hot矢量)   controlInputs = [2,5,3,4] # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight), 比如[0 1 0 0]表示 Straight
        - input_data[1]: 测量量input_speed（一个维度）
 - targets：维度为2的列表
      - targets[0]: 一维表示target_speed(维度为1)
      - targets[1]: 一维表示target_control（维度为3）
## 引用函数
 - controlOpTensors = controlNet(inputs, targets, shapeInput, dout, branchConfig, params, scopeName='controlNET')
### 函数-- controlNet
controlOpTensors = controlNet(inputs, targets, shapeInput, dout, branchConfig, params, scopeName='controlNET')

#### 输入：
 - inputs
 - targets
 - shapeInput
 - dout
 - branchConfig
 - params
 - scopeName='controlNET'
#### 输出：
 - controlOpTensors= {
    - 'optimizers': contSolver, 
    - 'losses': contLoss, 总的损失值
    - 'output': networkTensor, 网络的输出值
    - 'print': pr % 应该表示的是命令}
#### 主要内容：
 - networkTensor = load_imitation_learning_network(inputs[0], inputs[1],shape[1:3], dropoutVec)
    - networkTensor： 返回值是和branchConfig个数相关的量；networkTensor的维度是和branchConfig维度一样; networkTensor 是Branch网络的输出值
 - 对比网络的输出networkTensor和标签值targets可以得到误差means：
      - [steer gas brake]类型的Branch的输出为：
           - part = tf.square(tf.subtract(networkTensor[i], targets[1]))
      - [speed]类型的Branch的输出为：
           - part = tf.square(tf.subtract(networkTensor[-1], targets[0]))
      - means = tf.convert_to_tensor(parts)
 - 根据输入的控制命令需要创建一个mask
      - mask = tf.convert_to_tensor(inputs[1][0])
 - 总的损失函数值为contLoss


#### 函数---load_imitation_learning_network
networkTensor = load_imitation_learning_network(inputs[0], inputs[1],shape[1:3], dropoutVec)
##### 输入：
 - inputs[0] 输入图像shape=(?, 88, 200, 3)
 - inputs[1] 输入的控制指令和测量量 
      - input_data[0]: 一维表示输入的控制量input_control (维度为4)
      - input_data[1]: 测量量input_speed（一个维度）
 - shape[1:3] 图像大小[88, 200]
 - dropoutVec shape=(23,)
##### 输出;
 - networkTensor： 返回值是和branchConfig个数相关的量；networkTensor的维度是和branchConfig维度一样。
     - 比如： branchConfig = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]
     那么networkTensor是维度为4的列表，因为branchConfig的元素都是三维的，因此networkTensor列表的每一个元素也都是三维的。

##### 引用函数
 - network_manager = Network(dropout, tf.shape(x))
##### 主要内容：构建网络结构
- 输入图像88*200*3---经卷积和全连接层输出512维矢量 x
- 输入speed ---经全连接层输出128维矢量 speed
- 将两种输入量结合在一起，用变量j表示：j = tf.concat([x, speed], 1)
- 变量j----经全连接层输出512维矢量
- Branch网络:
  - branch_config:四个命令[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]
    - 若branch_config=["Steer", "Gas", "Brake"] 则输入j输出256维量
    - 若branch_config="Speed" 则输入x输出256维量

##### 函数---Network 函数提供了用于构建网络各层函数
network_manager = Network(dropout, tf.shape(x))
###### 输入
 - dropout 23维初始化矢量
 - tf.shape(x) 图像形状
###### 输出
 - network_manager

# genBranch函数
genBranch(fileNames=datasetFilesTrain, branchNum=3, batchSize=200):
- 输入： fileNames，branchNum=3（第几个Branch）, batchSize=200
- 输出： batchX, batchY
     - batchX维度：(batchSize, 88, 200, 3)
     - batchY维度：(batchSize, 28)
     
