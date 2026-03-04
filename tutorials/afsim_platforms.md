# AFSIM 平台与类型定义教程

## 1. 基本概念 (Definitions)
类 (Class / 平台类型)：用于创建对象的模板，包含特定的属性信息 。
继承 (Inheritance)：允许一个类由另一个类定义，在保留父类属性的同时进行修改或分配唯一特征 。不同型号的平台（如 F/A-18 系列或全球鹰系列）可通过继承派生来提升性能或增加载荷 。
实例 (Instance / 平台)：类的具体物理实现，所有仿真中的实体都是特定类型的实例 。
方法 (Method)：与对象关联的过程，用于访问或操作平台信息，如速度、航向和可用武器 。

## 2. 命名约定 (Naming Conventions) 
在编写脚本和定义对象时应遵循以下规范：
通用名字：使用连字符 - 或下划线 _ 替换空格或特殊字符 。
平台类型与部件对象类：采用全部大写字母，并用下划线替换空格、斜杠或破折号 。
对象实例化 (具体平台)：采用全部小写字母 。
文件名：采用小写字母 。
内部定义名称：应与文件名一致（不含后缀）且全部大写 。
系统组件：通常以所属系统的名称作为前缀 。

## 3. 平台构成与基本组件 
平台是仿真的主要实体，其定义包含信息属性与物理/逻辑组件：
信息和属性：包含名称、阵营 (Side)、信号特征 (IR/光学/RF)、指挥链以及航迹列表等 。
五大核心组件：
移动器 (Mover/推进器)：定义平台的运动学模型 。
传感器 (Sensor)：定义平台如何感知环境 。
武器 (Weapon)：定义平台对其他实体的物理影响 。
通信 (Comm)：定义平台间的信息交换方式 。
处理器 (Processor)：定义平台的逻辑控制与行动脚本 。

## 4. 推荐的项目文件目录结构 
建议将项目模块化以提高可维护性：
projectName.txt：主控文件，包含仿真时间和全局控制 。
setup.txt：配置总览文件，用于包含各功能模块 。
platforms/：存放平台类型 (platform_type) 定义文件 。
scenarios/：存放平台布局 (platform) 及位置设置文件 。
sensors/ / weapons/ / signatures/ / processors/：分别存放相应的子组件独立定义 。
output/：用于存储运行时生成的日志、事件记录及仿真结果文件 。

## 5. 脚本编码示例

### 5.1 平台类型定义 通常为每种类型创建独立文件，再由 setup.txt 统一引用 。
```afsim
platform_type TANK WSF_PLATFORM
    icon tank
end_platform_type

platform_type BOMBER WSF_PLATFORM
    icon bomber
end_platform_type
```

### 5.2 平台实例定义 定义实例时，名称使用小写，类型引用使用大写 。
```afsim
platform ship SHIP
    position 30:11:07.440n 80:49:32.604w
    side red           // 定义阵营 
    icon carrier       // 更改平台图标
end_platform

platform bomber BOMBER
    position 30:00:03.699n 80:30:19.006w
    side red
    altitude 30000 ft  // 设置初始高度 
    heading 270 deg    // 设置初始航向
end_platform
```

### 5.3 仿真控制与引用 
```afsim
include_once setup.txt   // 包含配置文件。若文件不存在则会报打开错误 

log_file output/mission.log  // 设置日志保存路径 

event_output
    file output/mission.evt  // 设置事件文件保存路径 
end_event_output

event_pipe
    file output/mission.aer  // 设置可视化仿真数据保存路径 
end_event_pipe

end_time 1 hr            // 设置仿真时长（默认为1分钟） 
```