# AFSIM 平台与类型定义教程

## 1. 基本概念
类（Class）是创建对象的模板，包含特定的属性信息（即平台类型 platform_type）。
实例（Instance）是类的特定实现（即平台 platform）。
继承允许子平台类型从父平台类型继承属性。

## 2. 命名约定
在 AFSIM 脚本中建议遵循以下命名约定：

| 对象类型 | 约定 | 示例 |
| :--- | :--- | :--- |
| 平台类型 (Platform Type) | 全大写，用下划线替换空格 | `F-18_RADAR`, `WSF_PLATFORM` |
| 平台实例 (Platform Instance) | 全小写 | `f-18_1`, `ship`, `tank_1` |
| 文件名 | 小写 | `generic_f-18_radar.txt` |

## 3. 文件结构
建议使用 `include_once` 命令来组织文件，而不是将所有代码写在一个文件中。
典型的项目结构如下：
- `setup.txt`: 定义所有的平台类型 (Platform Types)。
- `platforms/`: 存放具体平台实例定义的文件夹。
- `scenarios/`: 存放具体场景布局 (Laydown) 的文件。

## 4. 脚本语法示例

### 定义平台类型 (setup.txt)
平台类型通常在 `setup.txt` 中定义，使用 `platform_type` 和 `end_platform_type` 关键字。

```afsim
// 定义轰炸机类型
platform_type BOMBER WSF_PLATFORM
    icon bomber
end_platform_type

// 定义坦克类型
platform_type TANK WSF_PLATFORM
    icon tank
end_platform_type

// 定义舰船类型
platform_type SHIP WSF_PLATFORM
    icon ship
end_platform_type
```

### 定义平台实例 (platform definition)
具体的平台实例通常在场景文件（如 scenarios/blue_laydown.txt）中定义。 注意：定义平台时需要指定位置 (position)、阵营 (side) 和航向 (heading) 等属性。

```afsim
// 定义一艘红方舰船
platform ship SHIP
    position 30:23:20.040n 80:55:06.960w
    side red
    icon carrier
end_platform

// 定义一架红方轰炸机
platform bomber BOMBER
    position 30:16:00.120n 80:54:03.600w
    side red
    altitude 30000 ft
    heading 270 deg
end_platform

// 定义蓝方坦克 (注意单位和坐标格式)
platform tank_1 TANK
    position 30:22:43.680n 81:33:32.760w
    side blue
    heading 90 degrees
end_platform
```