# AFSIM 武器与执行命令 (Weapons and Execution)

## 1. 武器类型与组件结构

### 1.1 武器类型定义

* 显式武器 (Explicit Weapon)：会离开发射平台（如导弹），需要一个可发射的平台类型定义。
* 隐式武器 (Implicit Weapon)：不会离开发射平台（如干扰机、激光），作为平台部件附着。
* 预定义类型示例：WSF_EXPLICIT_WEAPON, WSF_RF_JAMMER, WSF_LASER_WEAPON。

### 1.2 武器定义 (Weapon Definition Block) 结构

一个完整的武器系统定义（例如空对空导弹）通常包含三个主要部分：

1.  被发射平台描述 (Launched Platform Description)：定义导弹自身的平台类型。
2.  武器效果 (Weapon Effect)：定义武器对目标造成的影响（杀伤力）。
3.  武器块 (Weapon Definition Block)：聚合上述组件并定义武器对象类。

---

## 2. 步骤 1：定义被发射平台（导弹）

导弹本质上是一个具有特殊行为的平台类型。它需要定义自己的运动、追踪和引信。

### 语法示例：`ATA-MISSILE` 平台类型

```afsim
// 步骤 1：定义导弹的平台类型
platform_type ATA-MISSILE WSF_PLATFORM
    radar_signature ATA_MISSILE_SIG
    icon missile
    
    // 移动器：用于飞行特性
    mover WSF_STRAIGHT_LINE_MOVER
        // 飞行时间/速度曲线 (示例)
        update_interval 0.5 s
        tof_and_speed
            0.0 1700 kts
            20.0 1500 kts
            40.0 1300 kts
        end_tof_and_speed
        maximum_lateral_acceleration 9.0 g
        guidance_mode lead_pursuit
    end_mover
    
    // 处理器：用于追踪目标
    processor tracker WSF_PERFECT_TRACKER
        update_interval 1.0 s
    end_processor
    
    // 处理器：用于引爆/终止
    processor fuse WSF_AIR_TARGET_FUSE
        max_time_of_flight_to_detonate 60 s
    end_processor
    
end_platform_type
```

## 3. 步骤 2：定义武器效果 (Weapon Effect)
武器效果定义了武器的杀伤效力（Lethality）。它通过 weapon_effects ... end_weapon_effects 块定义，通常派生自预定义的杀伤类型。

语法示例：球形杀伤力 (WSF_SPHERICAL_LETHALITY)
代码段

```afsim
// 步骤 2：定义武器的杀伤效果
weapon_effects ATA-MISSILE-EFFECTS WSF_SPHERICAL_LETHALITY
    // 定义杀伤半径范围
    minimum_radius 1000 m
    maximum_radius 1200 m
end_weapon_effects
```

预定义杀伤力类型
WSF_SPHERICAL_LETHALITY：球形杀伤力（基于距离和指数）。

WSF_GRADUATED_LETHALITY：分级杀伤力（基于距离和概率）。

其他类型包括：WSF_CARLTON_LETHALITY, WSF_EXOATMOSPHERIC_LETHALITY, WSF_HEL_LETHALITY 等。

## 4. 步骤 3 & 4：创建武器块并加载到平台
### 4.1 创建武器对象类 (Weapon Definition Block)
武器块 (weapon ... end_weapon) 聚合了导弹类型和武器效果，创建了一个可被平台携带的武器对象类。

代码段

```afsim
// 步骤 3：定义武器对象类 (GBU-39)
weapon GBU-39 WSF_EXPLICIT_WEAPON
    // 指定发射的平台类型（即导弹本身）
    launched_platform_type GBU-39 
    
    // 指定武器效果
    weapon_effects GBU-39_EFFECTS
    
    // 携带数量（可选，定义了平台初始库存）
    quantity 2 
end_weapon
```

### 4.2 隐式武器示例（干扰机）
隐式武器（如干扰机）派生自 WSF_RF_JAMMER，不指定 launched_platform_type。

代码段

```afsim
// 干扰机武器示例
weapon SOJ_VHF_JAMMER WSF_RF_JAMMER
    maximum_number_of_spots 2
    slew_mode fixed
    maximum_range 200 km
    
    // 干扰机需要发射机（Transmitter）组件
    transmitter
        antenna_pattern SOJ_VHF_ANTENNA
        frequency_band 30 mhz 300 mhz
        power 100.0 w
        internal_loss 0.0 db
    end_transmitter
end_weapon
```

### 4.3 加载武器到发射平台类型
将武器对象类作为组件添加到发射平台类型中：

代码段

```afsim
// 步骤 4：将武器加载到轰炸机平台类型
platform_type BOMBER WSF_PLATFORM
    icon bomber
    mover WSF_AIR_MOVER
    end_mover
    
    // 加载 GBU-38，初始数量 2
    weapon gbu-38 GBU-38
        quantity 2 
    end_weapon
    
    // 加载 GBU-39，初始数量 6
    weapon gbu-39 GBU-39
        quantity 6
    end_weapon
    
end_platform_type
```

## 5. 平台航迹与执行指令
### 5.1 平台航迹 (track)
track 块定义了平台对另一对象的初始感知信息（或称航迹）。

位置互斥：position、range 或 bearing 只能指定其中之一。

重复定义：一个平台可以指定多个 track 块。

语法示例：初始化轰炸机对坦克的航迹
代码段

```afsim
platform bomber BOMBER
    // ... 其他定义 ...
    
    // 声明轰炸机已知悉 tank_1 和 tank_2 的航迹
    track platform tank_1 end_track
    track platform tank_2 end_track
    
    // ... 航线定义 ...
end_platform
```

### 5.2 执行指令 (execute)
execute 块用于定义一个脚本（Warlord Script），使其在指定时间或以指定间隔执行。

execute at_time <time> absolute：在指定的仿真时间点执行一次。

execute at_time <time> relative：在平台创建后的指定相对时间执行一次（仅在平台上下文内有效）。

execute at_interval_of <time>：以指定的时间间隔重复执行。

示例：定时发射武器
代码段

```afsim
platform_type BOMBER WSF_PLATFORM
    // ... Mover 和 Weapon 定义 ...
    
    // 在仿真开始后 100 秒执行发射指令
    execute at time 100 sec absolute
        // 发射 2 枚 gbu-38，目标是 MasterTrackList 中的第一个目标 (索引 0)
        Weapon("gbu-38").FireSalvo (MasterTrackList().TrackEntry(0), 2);
        
        // 发射 2 枚 gbu-39，目标是 MasterTrackList 中的第二个目标 (索引 1)
        Weapon("gbu-39").FireSalvo (MasterTrackList().TrackEntry(1), 2);
    end_execute
    
end_platform_type
```