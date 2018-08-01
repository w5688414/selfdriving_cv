

- data.hdf5: 
   - rgb:200(数量)*88*200*3(图片参数)
   - target:200*28
     - 有用数据： 
          - 0 steer   # 输出标签 
          - 1 gas     # 输出标签
          - 2 brake   # 输出标签
          - 11 speed  # 输入标签、中间量
          - High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)  # branch
     - 其余数据目前未使用。 # github issue 有解释
- paper中一处error：
Methodology--A Network Architecture 第一段中:
表述network输出2维 其实是3维
- input: 
   - image
   - measurement # 主要就是speed
   - commend 
      - branch  # 4个branch 存在一个default 0
         - 2 follow lane
         - 3 left
         - 4 right
         - 5 straight
         - 每个branch输出标签相同
- output
     - 0 steer   
     - 1 gas  
     - 2 brake
- 遇到异常情况，e.g.:本不该停车时停车
   - 使用 speed prediction branch 公式 # 重新计算油门 
- 函数调用
   - driving_benchmark.py  # line 227
   - agent.run_step() # 返回 steer,gas, break
   - run_step()  # line 89
      - 调用 imitation_learning.py 中_computer_action()
   - _compute_action()  # line 96
      - 调用_control_function()
      - 通过control_input 去选择branch  # line 145
      - 异常情况【_avoid_stopping】进入 speed branch # 主要是转弯的时候   # line 164
         

