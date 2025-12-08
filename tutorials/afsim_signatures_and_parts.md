# AFSIM 特征与组件管理器 (Signatures and Part Manager)

## 1. 特征 (Signatures) 概述

* **定义**：平台特征是平台对象上的属性，用于定义平台的各种空间和电子属性。
* **平台部件**：平台包含传感器、处理器、移动器和通信组件等部件。
* **特征属性**：特征可以定义为：
    * **空间属性**：方位角（水平面角度），仰角（垂直面角度）。
    * **电子属性**：频率、极化（电波方向）。
    * **状态**：例如默认状态（default）或发射状态（launching）。
* **坐标系**：AFSIM 使用平台相对坐标系：
    * 方位角范围为 -180 度到 180 度，其中 $0^{\circ}$ 是平台的前方。
    * 仰角范围为 -90 度到 90 度。
* **引用方式**：特征可以在平台内定义，也可以在外部定义后在平台中引用。

## 2. 特征定义语法

### 2.1 简单恒定特征 (Constant Signature)
常数值特征适用于所有方位、频率和极化。

```afsim
// 示例：定义一个恒定雷达特征 (10 dbsm)
radar_signature CUEBALL_10DB
    constant 10 dbsm
end_radar_signature
```

* 允许定义随方位角/仰角变化的复杂特征，系统会在数据点之间进行插值。
* 表定义需要提供：单位、各维度尺寸（方位和高程）。
* 方位坐标沿表向下延伸，高程坐标横跨该表。
* 每个方位和仰角对对应单个数据值。

```afsim
// 示例：使用内联表定义轰炸机雷达特征 (RCS)
radar_signature BOMBER_RADAR_SIG
    state default
        inline_table dbsm 24 5 // dbsm 为单位，24x5 为方位x仰角尺寸
            // 高程坐标 (横跨表格): -90.0 0.0 30.0 90.0
            -90.0 0.0 30.0 90.0

            // 方位角 -180.0 处的数据值
            -180.0 20.0 20.0 0.0 20.0 20.0
            // ... 更多数据点 ...
            90.0 40.0 20.0 20.0 20.0
        end_inline_table
    end_state
end_radar_signature
```

### 2.3 特征状态和波段定义
可以为特征定义多个状态（state），并使用脚本进行切换。也可以定义特定的频率特性（band）。

```afsim
infrared_signature ir_states
    state default
        band medium
            constant 10 w/sr
        band default
            constant 20 w/sr
    end_state
    state launching // 发射状态
        constant 50 w/sr
    end_state
end_infrared_signature
```

---

## 3. 平台特征的加载与引用

### 3.1 外部文件定义 (`common.txt`)
共享的特征和组件通常定义在一个公共文件（如 `common.txt`）中。

```afsim
// common.txt 示例
radar_signature VEHICLE_RADAR_SIGNATURE
    constant 1 m^2
end_radar_signature

infrared_signature VEHICLE_INFRARED_SIGNATURE
    constant 10 watts/steradian
end_infrared_signature

filter FILTER_TACTICS WSF_KALMAN_FILTER
    range_measurement_sigma 50.m
    bearing_measurement_sigma 0.1 deg
    // ... 其他参数 ...
end_filter
```

### 3.2 平台类型引用 (CAR 示例)
要在平台上使用外部特征，需要在平台类型定义中通过关键字引用其名称。

```afsim
// platforms/car.txt 示例
platform_type CAR WSF_PLATFORM
    icon car
    
    // 引用在 common.txt 中定义的特征
    infrared_signature VEHICLE_INFRARED_SIGNATURE
    optical_signature VEHICLE_OPTICAL_SIGNATURE
    radar_signature VEHICLE_RADAR_SIGNATURE
    
    mover WSF_GROUND_MOVER
    end_mover
end_platform_type
```

### 3.3 包含外部特征文件 (BOMBER 示例)
如果特征定义在单独的文件中，需要使用 `include_once` 命令将其包含进来。

```afsim
// platforms/bomber.txt 示例
include_once weapons/agm/gbu-38.txt
include_once signatures/bomber_infrared_sig.txt
include_once signatures/bomber_radar_sig.txt

platform_type BOMBER WSF_PLATFORM
    icon bomber
    
    // 引用包含进来的特征
    infrared_signature BOMBER_INFRARED_SIG
    radar_signature BOMBER_RADAR_SIG
    
    mover WSF_AIR_MOVER
    end_mover
    // ... 武器定义 ...
end_platform_type
```

---

## 4. 平台特征默认值

当未明确定义特征时，AFSIM 会使用默认值，但这通常会触发警告。

| 特征类型 | 默认值 |
| :--- | :--- |
| **声学** | 1kHz时100 dB-20uPa |
| **红外线** | $1000 w/sr$ |
| **光学** | $1000 m^2$ |
| **雷达** | $1000 m^2$ |
| **固有对比度** | 0.5 (红外传感器) |
```