# selfdriving_cv
this is the repository of deep learning for self driving

# linux tutorials
Ubuntu14.04下搜狗输入法安装：
https://blog.csdn.net/u011006622/article/details/69281580

记录：Ubuntu14.04 下 chrome的安装过程: https://blog.csdn.net/m0_37676373/article/details/78616715

Ubuntu 14.04右键终端的设置：https://www.linuxidc.com/Linux/2014-04/100498.htm

# environment
```
ubuntu 14.04
tensorflow
carla
python 2.7 (python 3.x not test)
```

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
# train model
```
sudo pip install keras
sudo pip install imgaug
sudo pip install opencv-python
python train.py
```

## data instruction
### input
images,measurements,commdand

images: data.rgb

measurements: 
```
    targets[:,8]---Brake Noise, float
    targets[:,9]---Position X, float
    targets[:,10]---Position Y, float
    targets[:,10]---Speed, float
    targets[:,11]---Collision Other, float
    targets[:,12]---Collision Pedestrian, float
    targets[:,13]---Collision Car, float
    targets[:,14]---Opposite Lane Inter, float
    targets[:,15]---Sidewalk Intersect, float

    targets[:,21]---Orientation X, float
    targets[:,22]---Orientation Y, float
    targets[:,23]---Orientation Z, float
```
command: 
```
    targets[:,0]---Steer, float 
    targets[:,1]---Gas, float
    targets[:,2]---Brake, float
    targets[:,3]---Hand Brake, boolean
    targets[:,4]---Reverse Gear, boolean
    targets[:,5]---Steer Noise, float
    targets[:,6]---Gas Noise, float
    targets[:,7]---Brake Noise, float
    targets[:,24]---High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
```

### parameter
    targets[:,19]---Platform time, float
    targets[:,20]---Game Time, float
    targets[:,24]---Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) )
    targets[:,25]---Camera (Which camera was used)
    targets[:,26]---Angle (The yaw angle for this camera)

### output
action: steering angle, acceleration
```
    targets[:,16]---Acceleration X,float
    targets[:,17]---Acceleration Y, float
    targets[:,18]---Acceleration Z, float
```


# reference
[CARLA Documentation][1]

[Conditional Imitation Learning at CARLA][2]

[1]:https://carla.readthedocs.io/en/latest/
[2]:https://github.com/carla-simulator/imitation-learning
