# selfdriving_cv
this is the repository of deep learning for self driving

# linux tutorials
Ubuntu14.04下搜狗输入法安装：
https://blog.csdn.net/u011006622/article/details/69281580

## install carla python
```
cd PythonClient/
sudo python setup.py install
```
# instructions
```
sudo pip install tensorflow
sudo pip install scipy
sudo pip install numpy=1.14.5
./CarlaUE4.sh -windowed -ResX=640 -ResY=480
```
# run model
```
python run_CIL.py
./CarlaUE4.sh -windowed -ResX=640 -ResY=480 -carla-server
```
# reference
[CARLA Documentation][1]

[Conditional Imitation Learning at CARLA][2]

[1]:https://carla.readthedocs.io/en/latest/
[2]:https://github.com/carla-simulator/imitation-learning
