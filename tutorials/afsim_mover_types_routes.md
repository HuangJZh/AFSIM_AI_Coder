# AFSIM 运动类型与航线 (Movers and Routes)

## 1. 运动器 (Mover) 概述
在 AFSIM 中，无组件的平台（无运动器、传感器等）是静止的，且无法相互影响。必须将运动器组件添加到平台类型定义中，平台才能在仿真中执行移动操作。选择运动器时，应当选择与平台物理特性相符的类型。

## 2. 预定义运动器类型分类

### 2.1 航线类型 (Route Types)
用于定义带有航点（waypoints）的航线，使其按预定路线移动：
WSF_AIR_MOVER：适用于空中载具。
WSF_GROUND_MOVER：适用于在地面平原移动的坦克、汽车等。
WSF_ROAD_MOVER：用于沿道路移动。注意：使用此类型需要预先加载道路网数据。
WSF_SURFACE_MOVER：适用于舰船等水面平台。该类型支持负的高度值，可用于模拟水下潜艇的深度。

### 2.2 跟随者类型 (Follower Types)
这些运动器通常附加在具有航线运动器的平台之后，使该平台跟随其他平台移动：
WSF_HYBRID_MOVER：混合跟随模式。
WSF_OFFSET_MOVER：偏移跟随模式。

### 2.3 卫星/轨道类型 (Satellite/Orbit Types)
用于模拟地外轨道运行：
WSF_SPACE_MOVER：基础空间运动器。
WSF_NORAD_SPACE_MOVER：需要双线元素（TLE）格式的数据来定义轨道。

## 3. 向平台类型添加运动器
运动器组件通过 mover ... end_mover 块添加到平台类型 (platform_type) 定义中。

平台类型	适用的运动器示例	                代码块示例
空中平台	WSF_AIR_MOVER	      mover WSF_AIR_MOVER ... end_mover
陆地平台	WSF_GROUND_MOVER	  mover WSF_GROUND_MOVER ... end_mover
水面平台	WSF_SURFACE_MOVER	  mover WSF_SURFACE_MOVER ... end_mover
空间平台	WSF_SPACE_MOVER	      mover WSF_SPACE_MOVER ... end_mover

## 4. 核心运动器指令详解

### 4.1 WSF_AIR_MOVER 核心指令
该运动器主要用于水平面的连续运动。它基于最大限制设置移动特性，但不模拟垂直方向的过渡性俯仰率，因此垂直高度转换是不连续的。
常用限制指令：
maximum linear acceleration：最大线性加速度。
maximum radial acceleration：最大径向加速度。
maximum altitude / minimum altitude：最大/最小高度限制。
roll rate limit / turn_rate_limit：滚转率与转弯率限制。
状态指令：altitude 和 speed。

### 4.2 WSF_STRAIGHT_LINE_MOVER (导弹专用)
用于模拟导弹等高速直线运动的目标，通常需要配合速度剖面使用。
tof_and_speed 块：定义飞行时间（Time of Flight）与对应速度的曲线。
语法规范：在 tof_and_speed 块中，时间数值后必须明确带上单位（如 s），否则会导致编译器报错。
正确示例：10.0 s 1500 kts
错误示例：10.0 1500 kts

## 5. 空间运动器 (WSF_SPACE_MOVER) 的特殊语法规则
在 WSF_SPACE_MOVER 的定义中，高度 (altitude) 不能作为独立的顶级指令出现。
错误写法：在 mover 块中直接使用 altitude 800 km 会触发 Unknown command 致命错误。
正确写法：高度必须作为 position 指令的参数写在同一行内。
```afsim
mover WSF_SPACE_MOVER
    position 0:00:00.000n 100:00:00.000e altitude 800 km
    heading 0 deg
end_mover
```

## 6. 航线 (Routes) 定义与逻辑
航线是直接添加到平台实例 (platform) 中的，用于定义一系列航点。

### 6.1 航线基本指令
label <名称>：为当前位置点设置一个标签名，方便后续跳转。
position <坐标> [altitude <高度>]：定义航点的地理位置。
speed <速度>：定义到达该航点时的期望速度。
goto <标签名>：在航路末尾指示平台返回到标记的标签位置，常用于创建死循环的巡逻路径。

### 6.2 巡逻航线示例
```afsim
platform ship SHIP
    position 30:23:20n 80:55:06w
    side red
    route
        label Start
        position 30:23:20n 80:55:06w altitude 0.0 ft
        speed 30 nm/h 
        position 30:30:07n 80:54:49w altitude 0.0 ft
        position 30:28:38n 80:48:17w altitude 0.0 ft
        goto Start // 循环执行巡逻
    end_route
end_platform
```