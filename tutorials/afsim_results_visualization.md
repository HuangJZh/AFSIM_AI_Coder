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

