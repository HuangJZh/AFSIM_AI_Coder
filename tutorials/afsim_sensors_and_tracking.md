# AFSIM 传感器与航迹 (Sensors and Tracking)

## 1. 坐标系约定 (Coordinate System Conventions)

### 1.1 世界坐标系 (World Coordinate System)
* 基准：地球表面建模为 WGS-84 标准定义的扁椭球体。
* 系统：右手笛卡尔系统。
* 原点：在地球中心。
* 轴向：+X轴通过 0N, 0E；+Y轴通过 0N, 90E；+Z轴穿过 90N (北极)。

### 1.2 实体坐标系 (Entity Coordinate System, ECS)
* 原点：在实体的质心处。
* 系统：右手笛卡尔系统。
* 轴向：+X轴位于实体的前面；+Y轴位于实体的右侧；+Z轴位于实体的底部。
* 旋转：
    * 偏航 (Yaw)：绕 Z 轴旋转，正向使机头向右。
    * 俯仰 (Pitch)：绕 Y 轴旋转，正向使机头向上。
    * 滚转 (Roll)：绕 X 轴旋转，正滚使右翼下降。

---

## 2. 传感器参数与范围

### 2.1 传感器范围定义
* 回转极限 (Gimbal Limits)：定义铰接部件相对于 ECS 可回转的范围。
* 扫描极限 (Scan Limits)：定义传感器模式中心左右的搜索覆盖区域。
* 视野极限 (Field of View Limits)：大于或等于扫描极限，用于忽略不在 field_of_view_limits 限制内的对象，使仿真运行速度更快。

### 2.2 定义天线方向图 (`antenna_pattern`)
天线方向图用于定义通信和传感器设备的增益。

```afsim
// 示例：矩形天线方向图
antenna_pattern ACQ_RADAR_ANTENNA rectangular_pattern
    peak_gain 35 dB
    minimum_gain -10 db
    azimuth_beamwidth 10 deg
    elevation_beamwidth 10 deg
end_rectangular_pattern
end_antenna_pattern
```

雷达传感器定义了其检测能力、扫描模式和收发机配置。

```afsim
sensor ACQ_RADAR WSF_RADAR_SENSOR
    // 检测能力
    one_m2_detect_range 50 nm     // 1平方米目标检测范围
    maximum_range 150 nm          // 最大跟踪范围
    frame_time 10 sec             // 帧时间（决定报告时间）
    antenna_height 5 m
    
    // 扫描模式
    scan_mode azimuth_and_elevation
    azimuth_scan_limits -180 deg 180 deg
    elevation_scan_limits 0 deg 50 deg
    
    // 发射机 (Transmitter) - 调整噪声以匹配指定的检测范围
    transmitter
        antenna_pattern ACQ_RADAR_ANTENNA // 引用已定义的天线
        power 1000 kw
        frequency 3000 mhz
        internal_loss 2 db
    end_transmitter
    
    // 接收机 (Receiver)
    receiver
        antenna_pattern ACQ_RADAR_ANTENNA
        bandwidth 2 mhz
        noise_power -160 dbw // 将根据 1 m^2 进行校准
        internal_loss 7 dB
    end_receiver
    
    // 跟踪要求
    probability_of_false_alarm 1.0e-6
    required_pd 0.5
    swerling_case 1
    hits_to_establish_track 5 3   // 建立航迹所需的命中数
    hits_to_maintain_track 5 1
    track_quality 0.6             // 航迹质量阈值 (用于战术)
    
    // 报告内容（必须明确定义传感器发送的航迹信息）
    reports_range
    reports_bearing
    reports_elevation
    reports_iff
    
    end_sensor
```

---

## 3. 跟踪流程与组件管理器

### 3.1 跟踪处理器与任务处理器
* 传感器航迹 (Sensor Tracks)：传感器创建的航迹，也被称为“原始航迹” (`raw` tracks)。
* 跟踪处理器 (`WSF_TRACK_PROCESSOR`)：负责存储、融合航迹，并维护**主跟踪列表**。它也被称为“数据管理器” (`data_mgr`)。
* 任务处理器 (`WSF_TASK_PROCESSOR`)：负责读取主跟踪列表，并执行战术脚本。

### 3.2 平台组件连接 (`internal_link`)
平台内的组件（如传感器、通信模块、处理器）之间通过 `internal_link` 命令进行连接。

```afsim
platform_type SINGLE_LARGE_SAM WSF_PLATFORM
    // ...
    
    comm cmdr_net RED_DATALINK
        // 通信组件将数据发送给 data_mgr 处理器
        internal_link data_mgr
        internal_link task_mgr 
    end_comm
    
    sensor acq ACQ_RADAR
        on
        // 传感器将航迹通过 internal_link 发送给 data_mgr
        internal_link data_mgr 
    end_sensor
    
    processor data_mgr WSF_TRACK_PROCESSOR
        purge_interval 60 seconds
    end_processor
    
    processor task_mgr WSF_TASK_PROCESSOR
    end_processor
    
end_platform_type
```

### 3.3 平台预定义航迹 (`track`)
`track` 块定义了平台对另一对象在模拟开始时的初始感知信息。这些航迹在模拟开始时在主跟踪列表中启动。

```afsim
platform bomber BOMBER
    // ... 其他定义 ...
    
    // 声明轰炸机已知悉 tank_1 和 tank_2 的航迹
    track platform tank_1 end_track
    track platform tank_2 end_track
    
end_platform
```

### 3.4 事件输出 (`event_output`)
`event_output` 块用于定义在仿真过程中需要记录到事件日志 (`.evt` 文件) 中的事件类型。

```afsim
// event_output.txt
event_output
    enable LOCAL_TRACK_DROPPED    // 本地航迹丢失
    enable LOCAL_TRACK_INITIATED  // 本地航迹建立
    enable MESSAGE_RECEIVED       // 消息接收
    enable MESSAGE_TRANSMITTED    // 消息发送
    enable SENSOR_TRACK_DROPPED   // 传感器航迹丢失
    enable SENSOR_TRACK_INITIATED // 传感器航迹建立
end_event_output
```
```