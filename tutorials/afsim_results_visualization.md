# AFSIM 结果可视化 (Results Visualization)

## 1. 概述与核心文件

* **目的**：结果可视化工具用于在地理环境中显示记录的模拟结果，是进行后处理分析的工具。
* **数据来源**：可视化工具读取模拟产生的 **AER 文件**。
* **AER 文件生成**：AER 文件通过 `event_pipe` 块定义。
* **其他文件**：模拟结果通常还包括 `.evt` (事件输出) 和 `.log` (日志文件)。

### 1.1 场景文件配置 (`floridistan.txt` 示例)

```afsim
// floridistan.txt 示例
// ... 包含 setup.txt, laydown.txt 等文件 ...

log_file output/jacksonabad.log

event_output file output/jacksonabad.evt end_event_output
event_pipe file output/jacksonabad.aer end_event_pipe

end_time 1 hr
// #final run number 5 (可选)
```

`event_pipe` 块允许用户创建模拟事件的二进制记录。

```afsim
event_pipe file output/jacksonabad.aer
    use_preset full // 使用完整预设
    disable DRAW    // 禁用绘制事件（可选）
end_event_pipe
```

### 1.3 `event_pipe` 指令
* `use_preset [default | low | high | full]`：配置输出的事件类型。
* `enable/disable <事件组>`：指定需要包含或排除的事件组。

### 1.4 事件管道预设内容

| 预设 | 包含的事件类型 (BASE DATA, ENTITY STATE, DRAW, DETECTION_CHANGE, COMMENT, TRACK 始终默认包含) |
| :--- | :--- |
| **default / low** | - (仅默认包含项) |
| **high** | 默认包含项 + TRACK_UPDATE, MESSAGE_RECEIVED, MESSAGE_TRANSMITTED |
| **full** | high 包含项 + **DETECTION_ATTEMPT** |

---

## 2. 结果可视化界面

### 2.1 打开 AER 文件
* **方式 1**：在项目浏览器中双击 `.aer` 文件。
* **方式 2**：在模拟运行完成后的输出窗口中选择结果文件。
* **方式 3**：运行 `results_vis.exe`，在启动窗口选择最近记录或浏览文件。

### 2.2 可视化内容
可视化工具可显示多种结果，包括：
* **地理环境**：地图显示 (Map Display)。
* **平台**：平台细节 (Platform Details)，平台浏览 (Platform Browser)。
* **历史**：平台历史（航迹线 Tracelines 和翼带 Wing ribbons）。
* **传感器**：传感器探测区域，平台交互线 (Interaction Lines)。
* **数据分析**：图表、曲线图（例如 Altitude vs. Lifetime）、表格（例如武器作战数据）。

### 2.3 时间控制
时间轴控件用于回放模拟结果：
* **向前/向后播放**。
* **拖动**：调整模拟时间。
* **设置播放速率**。
* **仿真时钟 (Simulation Clock)**：显示当前模拟时间。
* **灰色区域**：显示当前内存中已加载的场景内容。

---

## 3. 分析数据类型

可视化工具包含了分析以下内容的工具：
* **平台数据**：如位置、速度、状态、高度等。
* **传感器数据**：如探测范围、传感器追踪 (Sensor Track) 和本地追踪 (Local Track)。
* **武器作战数据**：如武器发射时间、目标、剩余数量等。
* **数据导出**：可以将数据导入 `.csv` 文件进行后处理分析。
```