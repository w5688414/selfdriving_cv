## remote operations
- 服务器密码：t88888888

```
ssh ubuntu@117.50.22.59
上传本地文件到远程机器指定目录
scp -r yolo_9000/ ubuntu@117.50.22.59:/home/ubuntu/

scp -r /home/ydx/Desktop/Auto\ Vechicles/CORL2017ImitationLearningData.tar.gz ubuntu@117.50.22.59:/data


ssh ubuntu@117.50.22.59
查看云端文件
df -h

```

## UAI Train训练平台
```
sudo python tf_tool.py pack \
            --public_key=/HOySV2WKVkciASUmRP9dlLzVhiRTmSHz2mx9jHmmXdsehqAVrWOdA== \
			--private_key=f0d85113ac1f17ff822e2f63dc195109280982fd \
			--code_path=./code/ \
			--mainfile_path=mnist_summary.py \
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
sudo docker run -it -v /data/test/data/:/data/data -v /data/test/output/:/data/output test-cpu:uaitrain /bin/bash -c "cd /data && /usr/bin/python /data/mnist_summary.py --max_step=2000 --work_dir=/data --data_dir=/data/data --output_dir=/data/output --log_dir=/data/output"

上传镜像
sudo docker push uhub.service.ucloud.cn/trytrain/trainjx:uaitrain

上传数据集
./filemgr-linux64 --action mput --bucket datasets --dir /home/eric/self-driving/docker/uai-sdk/examples/tensorflow/train/mnist_summary_1.1/data --trimpath /home/eric/self-driving/docker/uai-sdk/examples/tensorflow/train/mnist_summary_1.1
```

## reference
[TensorFlow训练镜像打包][1]

[使用UAI Train训练平台][2]

[使用UFile管理工具上传下载数据][3]

[1]: https://docs.ucloud.cn/ai/uai-train/guide/tensorflow/packing
[2]: https://docs.ucloud.cn/ai/uai-train/tutorial/tf-mnist/train
[3]: https://docs.ucloud.cn/ai/uai-train/base/ufile/files

