## remote operations
- 服务器密码：t88888888

```
ssh ubuntu@117.50.22.59
或
ssh ubuntu@117.50.23.79

上传本地文件到远程机器指定目录
scp -r /home/ydx/Desktop/Auto\ Vechicles/CORL2017ImitationLearningData.tar.gz ubuntu@117.50.22.59:/data
从服务器上下载指定目录到本机：
scp -r  ubuntu@117.50.23.79:~/carla-train /home/eric/
查看云端文件
df -h

chmod –R 777 * :
参数-R : 对目前目录下的所有档案与子目录进行相同的权限变更(即以递回的方式逐个变更)

```
## tensorboard 使用
```
tensorboard --logdir=/tmp --port=6005
ssh -L 6005:127.0.0.1:6005 ubuntu@117.50.10.71
ssh之后去浏览器打开127.0.0.1:6005
```

## vim 命令
```
 yy    复制整行（nyy或者yny ，复制n行，n为数字）； 
 p      小写p代表贴至游标后（下），因为游标是在具体字符的位置上，所以实际是在该字符的后面 
 P      大写P代表贴至游标前（上） 
    整行的复制粘贴在游标的上（下）一行，非整行的复制则是粘贴在游标的前（后）
dd:删除游标所在的一整行(常用)
ndd:n为数字。删除光标所在的向下n行，例如20dd则是删除光标所在的向下20行
```

## tmux 命令
```
tmux new -s demo # 新建一个名称为demo的会话
tmux a -t demo # 进入到名称为demo的会话
Ctrl+b	d	断开当前会话
```
https://blog.csdn.net/chenqiuge1984/article/details/80132042

## ucloud 挂载云盘
```
 mount /dev/vde /tenplus
```
https://docs.ucloud.cn/storage_cdn/udisk/userguide/format/linux

## UAI Train训练平台
```
sudo python tf_tool.py pack \
            --public_key=/HOySV2WKVkciASUmRP9dlLzVhiRTmSHz2mx9jHmmXdsehqAVrWOdA== \
			--private_key=f0d85113ac1f17ff822e2f63dc195109280982fd \
			--code_path=./code/ \
			--mainfile_path=train.py \
			--uhub_username=476226078@qq.com \
			--uhub_password=cmbjxX666 \
			--uhub_registry=trytrain \
			--uhub_imagename=trainjx \
                        --internal_uhub=true \
			--ai_arch_v=tensorflow-1.1.0 \
			--test_data_path=/data/test/data \
			--test_output_path=/data/test/output \
			--train_params="--max_step=2000" \

创建docker
sudo docker build -t test-cpu:uaitrain -f uaitrain-cpu.Dockerfile .

本地运行
sudo docker run -it -v /data/test/data/:/data/data -v /data/test/output/:/data/output test-cpu:uaitrain /bin/bash -c "cd /data && /usr/bin/python /data/train.py --max_step=2000 --work_dir=/data --data_dir=/data/data --output_dir=/data/output --log_dir=/data/output"

上传镜像
sudo docker push uhub.service.ucloud.cn/trytrain/trainjx:uaitrain

上传数据集
./filemgr-linux64 --action mput --bucket datasets --dir /home/eric/self-driving/docker/uai-sdk/examples/tensorflow/train/mnist_summary_1.1/data --trimpath /home/eric/self-driving/docker/uai-sdk/examples/tensorflow/train/mnist_summary_1.1
```

## reference
[TensorFlow训练镜像打包][1]

[使用UAI Train训练平台][2]

[使用UFile管理工具上传下载数据][3]

[vim 删除一整块，vim 删除一整行][4]

[vi/vim复制粘贴命令][5]


[1]: https://docs.ucloud.cn/ai/uai-train/guide/tensorflow/packing
[2]: https://docs.ucloud.cn/ai/uai-train/tutorial/tf-mnist/train
[3]: https://docs.ucloud.cn/ai/uai-train/base/ufile/files
[4]: https://blog.csdn.net/chenyoper/article/details/78260007
[5]: https://blog.csdn.net/lanxinju/article/details/5727262