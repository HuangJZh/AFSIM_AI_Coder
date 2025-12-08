# AFSIM 运动类型与航线 (Movers and Routes)

## 1. 运动器 (Mover) 概述

无组件的平台不会移动或互相影响。
需要将运动器 (movers) 等组件添加到平台上，以便它们执行操作和移动。
定义平台时，应当选择与平台类型相符的移动器。

## 2. 预定义运动器类型分类 

AFSIM 中预定义的运动器类型可以分为以下几大类：

**航线类型 (Route Types)**：
    这类移动器用于定义带有航点 (waypoints) 的航线，使其在仿真过程中按预定路线移动。
    包括 `WSF_AIR_MOVER`、`WSF_GROUND_MOVER`、`WSF_ROAD_MOVER`、`WSF_SURFACE_MOVER`等。
**跟随者类型 (Follower Types)**：
    这些移动器附加到航线类型移动器上，用于使一个平台实例跟随其他平台。
    包括 `WSF_HYBRID_MOVER`和 `WSF_OFFSET_MOVER`。
**卫星/轨道类型 (Satellite/Orbit Types)**：
    需要一个双线元素 (TLE) 格式的数据来定义平台的轨道。
    包括 `WSF_NORAD_SPACE_MOVER`和 `WSF_SPACE_MOVER`。

## 3. 语法：向平台类型添加运动器

运动器组件通过 `mover ... end_mover` 块添加到平台类型 (`platform_type`) 定义中：

| 平台类型 | 适用的运动器 | 代码示例 |
| :--- | :--- | :--- |
| 空中平台 (如轰炸机)`WSF_AIR_MOVER` | 
```afsim
platform_type BOMBER WSF_PLATFORM
    icon bomber
    mover WSF_AIR_MOVER
    end_mover
end_platform_type
```
| 陆地平台 (如坦克/汽车) | `WSF_GROUND_MOVER`  | 
```afsim
platform_type TANK WSF_PLATFORM
    icon tank
    mover WSF_GROUND_MOVER
    end_mover
end_platform_type
``` 
|水面平台 (如舰船) | `WSF_SURFACE_MOVER` | 
```afsim
platform_type SHIP WSF_PLATFORM
    icon ship
    mover WSF_SURFACE_MOVER
    end_mover
end_platform_type
``` 
| 空间平台 (如卫星)  | `WSF_SPACE_MOVER`  | 
```afsim
platform_type SATELLITE WSF_PLATFORM
    icon satellite
    mover WSF_SPACE_MOVER
    end_mover
end_platform_type
```

## 4. WSF_AIR_MOVER 核心指令 

`WSF_AIR_MOVER` 是为简化空中载具运动设计的航线移动器 。它的移动基于设置的最大限制（例如线性加速度、径向加速度、速度、G值），但它仅适用于水平面的连续运动。它不模拟过渡性垂直俯仰率和垂直加速度，因此垂直过渡是不连续的。

主要的移动器命令 (Mover Commands) 包括：

`altitude`, `speed`
`maximum linear acceleration` , `maximum radial acceleration` 
`maximum altitude` , `minimum altitude`
`roll rate limit`, `turn rate limit`

## 5. 语法：定义航线 (Routes)

航线是添加到具体**平台实例**（Platform Instance）中的，用于定义一系列航点。

### 示例：给平台添加循环巡逻航线 

```afsim
// scenarios/red_laydown.txt 或 blue_laydown.txt
platform ship SHIP
    position 30:23:20.040n 80:55:06.960w
    side red
    icon carrier

    route // 航线定义开始
        // 航点 0 (起点, 可定义标签方便跳转)
        label Start 
        position 30:23:20.040n 80:55:06.960w altitude 0.00 ft 
        speed 30 nm/h // 初始速度：30节 

        // 航点 1
        position 30:30:07.579n 80:54:49.205w altitude 0.00 ft 

        // 航点 2
        position 30:28:38.694n 80:48:17.450w altitude 0.00 ft 

        // 航点 3 (Loop point)
        position 30:24:28.057n 80:48:21.408w altitude 0.00 ft 

        goto Start // 循环指令：在航线末尾返回到标记为 'Start' 的航点 
    end_route // 航线定义结束 
end_platform 