# 代码错误总结与修正

## 简单任务8

### 错误1
```afsim
mover WSF_SPACE_MOVER
    // 设置轨道参数
    altitude 800 km
    heading 0 deg
    position 0:00:00.000n 100:00:00.000e
end_mover
```
报错信息：ERROR: Could not find mover altitude 
altitude 通常不被识别为独立的指令。在 AFSIM 的坐标定义语法中，高度信息通常作为 position 命令的一个附属参数。正确的标准语法格式是将经纬度和高度写在同一行，即 position <纬度> <经度> altitude <高度>。
修正方案：
```
mover WSF_SPACE_MOVER
    // 设置轨道参数
    position 0:00:00.000n 100:00:00.000e altitude 800 km
    heading 0 deg        
end_mover
```


## 中等任务1

### 错误1
```afsim
hits_to_establish_track 5 3
hits_to_maintain_track 5 1
```
报错信息：ERROR: Expected value '3' to be >= 5 
在 AFSIM 传感器定义中，hits_to_establish_track 和 hits_to_maintain_track 的参数顺序是 <要求命中次数> <总尝试次数>。逻辑上，要求的命中次数必须小于或等于总尝试次数。
修正方案：
```afsim
hits_to_establish_track 3 5
hits_to_maintain_track 1 5
```

### 错误2
```afsim
mover WSF_AIR_MOVER
    altitude 30000 ft
    speed 400 kts
end_mover
```
报错信息：ERROR: Could not find mover altitude
platform_type 用于定义一类平台的“模板”或“属性类”，而不是具体的实体。在 AFSIM 的规范中，像 speed 和初始 altitude 这种描述具体实例物理状态的指令，不能出现在 platform_type 的定义中。这些具体的状态参数必须写在具体的平台实例（platform）声明中，或者在航线（route）的航点内定义。
修正方案：
```afsim
mover WSF_AIR_MOVER
end_mover
```

### 错误3
```afsim
//整段代码在platform_type FIGHTER内部
// 定义战斗机雷达特征
radar_signature FIGHTER_RADAR_SIG
    constant 5 m^2
end_radar_signature
// 应用雷达特征
radar_signature FIGHTER_RADAR_SIG
```
在 AFSIM 中，一般将共享的特征（Signatures）作为一个独立的代码块在外部（全局）进行定义，然后在 platform_type 内部通过名称进行调用/引用。初始代码不仅在 platform_type 内部定义了一个完整的块，紧接着又用同名指令去“应用”它，导致了语法解析的混乱。
修正方案：
```afsim
// 将特征定义移到全局（platform_type 外部）
radar_signature FIGHTER_RADAR_SIG
    constant 5 m^2
end_radar_signature
platform_type FIGHTER WSF_PLATFORM
    icon fighter
    mover WSF_AIR_MOVER
    end_mover
    // 仅在内部引用外部定义的特征
    radar_signature FIGHTER_RADAR_SIG
end_platform_type
```


## 中等任务3

### 错误1
代码开头定义了一个 platform_type BLUE_MLRS WSF_PLATFORM，随后在中间又完整定义了一次 platform_type BLUE_MLRS WSF_PLATFORM。
每一类 platform_type 定义只能出现一次。重复定义会导致编译器报重复声明的致命错误。
修正方案：
删除开头冗余/不完整的 BLUE_MLRS 定义，仅保留包含武器和执行脚本的完整定义。

### 错误2
```afsim
execute at_time 300 sec absolute
```
AFSIM 在执行块中不识别 sec 作为秒的单位，正确的指令语法必须严格缩写为 s。
修正方案：
```afsim
execute at_time 300 s absolute
```

### 错误3
```afsim
WsfTrack target = FindTrack("Target_1");
```
在 AFSIM 脚本语言中，并不存在全局直接调用 FindTrack() 这个内置函数来获取航迹的机制。要获取目标航迹，必须通过遍历当前平台的主航迹列表 (PLATFORM.MasterTrackList())，检查每个航迹的名称 (TargetName()) 来匹配目标。
修正方案：
```afsim
for (int i = 0; i < PLATFORM.MasterTrackList().Count(); i += 1)
{
    WsfTrack target = PLATFORM.MasterTrackList().TrackEntry(i);
    if (target.TargetName() == "Target_1")
    { ... }
}
```

### 错误4
```afsim
bool fired = Weapon("at_missile").FireSalvo(target, 1);
```
Weapon() 不是一个可以全局调用的游离函数，它是属于特定平台对象的成员方法。在 execute 脚本块中，必须显式指明上下文对象，即使用内置预定义变量 PLATFORM（代表当前平台自身）来调用。
修正方案：
```afsim
bool fired = PLATFORM.Weapon("at_missile").FireSalvo(target, 1);
```

### 错误5
没有在全局或场景布局文件中定义 Target_1。
如果在 blue_mlrs_1 的脚本和航迹感知的预设里尝试攻击 Target_1，那么 Target_1 必须作为具体的实例（platform）在代码中被明确定义出来。且由于 AFSIM 编译器的顺序解析要求，被追踪者（红方目标）必须定义在追踪者（蓝方发射车）的前面。
修正方案：
```afsim
// 必须定义在追踪者 blue_mlrs_1 之前
platform Target_1 WSF_PLATFORM
    position 25:05:00.000n 116:00:00.000e
    side red
end_platform
```

### 错误6
在 blue_mlrs_1 的实例中仅定义了位置和阵营等属性。
BLUE_MLRS 并没有装备雷达传感器，因此它本身无法靠“看”来感知 Target_1 的存在。要使得脚本里 MasterTrackList().Count() 不为空并能遍历到目标，就必须通过预定航迹感知 (track)指令人为赋予蓝方平台对红方目标的位置认知，否则它就“看不见”这个目标，也就无法开火。
修正方案：
```afsim
platform blue_mlrs_1 BLUE_MLRS
    // ...
    // 补充对预定目标的航迹感知
    track platform Target_1 end_track
end_platform
```


## 中等任务4

### 错误1
```afsim
// command_chain.txt
command_chain BLUE_AIR commander_air
command_chain BLUE_AIR wingman_air
```
command_chain 指令不能在全局直接游离使用。在 AFSIM 中，指挥链的从属关系必须在具体的平台实例（platform）内部进行声明。
修正方案：
```afsim
platform commander_air COMMANDER_AIR
    position 30:00:00.000n 80:00:00.000w
    altitude 25000 ft
    heading 0 deg
    command_chain BLUE_AIR SELF
end_platform

platform wingman_air WINGMAN_AIR
    position 30:01:00.000n 80:00:00.000w
    altitude 25000 ft
    heading 0 deg
    command_chain BLUE_AIR commander_air
end_platform
```


## 中等任务8

### 错误1
```afsim
// 假设 ATM-1 武器已定义并包含在 setup.txt 中
weapon atm-1 ATM-1
    quantity 10
end_weapon
```
编译器报错找不到武器 ATM-1。在 AFSIM 中，不能仅仅在注释中声明假设使用了某个武器，必须完整定义该武器的类型（包括武器的杀伤效果 weapon_effects、导弹平台类型 platform_type 以及明确的武器系统 weapon），才能在对应的平台类型中将其挂载。
修正方案：补充对ATM-1的完整定义。
```afsim
weapon_effects ATM_EFFECTS WSF_SPHERICAL_LETHALITY
    minimum_radius 5 m
    maximum_radius 15 m
end_weapon_effects

platform_type ATM_MISSILE_PLATFORM WSF_PLATFORM
    icon missile
    mover WSF_STRAIGHT_LINE_MOVER
    end_mover
end_platform_type

weapon ATM-1 WSF_EXPLICIT_WEAPON
    launched_platform_type ATM_MISSILE_PLATFORM
    weapon_effects ATM_EFFECTS
end_weapon

platform_type BLUE_TANK WSF_PLATFORM
    icon tank
    mover WSF_GROUND_MOVER
    end_mover

    weapon atm-1 ATM-1
        quantity 10
    end_weapon

    // ... 保留原有的 engage_logic 处理器 ...
end_platform_type
```


## 中等任务10

### 错误1
```afsim
inline_table dbsm 37 1
            -180.0
    -180.0   20.0
    -170.0   20.0
    // ...
```
在定义雷达散射截面（RCS）的二维内联表（inline_table）时，AFSIM 要求至少有两个高程（仰角）维度来进行插值（即列数必须 >= 2）。
修正方案：提供至少两个高程（仰角）列（例如 -90.0 到 90.0）并补全对应的数据。为了简洁和规范，可以参考标准示例建立一个 5x5 的内联表。
```afsim
radar_signature BOMBER_RADAR_SIG
    state default
        inline_table dbsm 5 5 
                  -90.0 -45.0 0.0 45.0 90.0            
            -180.0 10.0 10.0 10.0 10.0 10.0
            -90.0  15.0 15.0 15.0 15.0 15.0
            0.0    1.0  1.0  1.0  1.0  1.0
            90.0   15.0 15.0 15.0 15.0 15.0
            180.0  10.0 10.0 10.0 10.0 10.0
        end_inline_table
end_radar_signature
```