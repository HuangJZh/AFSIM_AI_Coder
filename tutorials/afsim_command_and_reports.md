# AFSIM 指挥链、报告和通信 (Command Chain, Reports and Communication)

## 1. 指挥链 (`command_chain`)

* **定义**：指挥链 (Command Chain) 是用于命令或报告的分层映射，指定给定平台的直接上级的平台子命令。
* **结构**：由指挥官、下级和同级组成。
* **指挥官**：每个链必须有一个指挥官（可以是 SELF，即平台自身）。
* **多重归属**：平台可以在多个命令链中，也可以在不同的链上有多个指挥官。
* **错误处理**：如果指挥官不存在，程序将执行，但 Wizard 会标记平台名称（不会产生编译错误）。

### 1.1 指挥链语法
```afsim
// 定义一个名为 PIRATE1 的指挥链，指挥官是 SELF
command_chain PIRATE1 SELF 

// 将另一个平台 (pirate_ac_1) 定义为 PIRATE1 链的指挥官
command_chain PIRATE1 pirate_ac_1
```

* **指挥官默认值**：`SELF`
* **命令链名称默认值**：`"default"`
* **链顶部判断**：脚本中可通过 `if( PLATFORM.Commander().Name() == PLATFORM.Name() )` 来判断当前平台是否为指挥链的顶部指挥官。

---

## 2. 通信设备 (`comm`)

* **通信设备**：是平台组件，允许平台与其他外部平台通信。
* **网络名称**：具有相同 `network_name` 的通信设备可以相互通信。
* **链路**：通信设备可以链接到平台上的其他部件（例如处理器）。

### 2.1 预定义通信类型
* **WSF_COMM_TRANSCEIVER** (Comm Transceiver)：用于完美的（有线）双向通信。
* **WSF_RADIO_TRANSCEIVER** (Radio Transceiver)：用于无线电双向通信，受视线限制。
* **WSF_JTIDS_TERMINAL**：内置的 Link-16 设备。
* **WSF_SUBSURFACE_RADIO_TRANSCEIVER**：允许水下通信。
* **收发器 (Transceiver)**：双向通信；RCVR (Receiver) 只接收；XMTR (Transmitter) 仅传输。

### 2.2 定义通信设备
```afsim
// 示例：定义一个无线电收发器，加入 pirate_radio 网络
comm pirate_comm WSF_COMM_TRANSCEIVER
    network_name pirate_radio
    internal_link data_mgr // 链接到平台上的 data_mgr 处理器
end_comm
```

### 2.3 内置指挥网络 (Local Networks)
AFSIM 提供特殊网络名称，基于 `commander-subordinate` 关系创建简单网络。

* **`<local:master>`**：表示“我是数据链上的指挥官”，用于指挥官侧。
* **`<local:slave>`**：表示“我是这个数据链的下属”，用于下级侧。
* **作用**：在指挥官及其下属之间创建通信网络。
* **注意**：不要在同一平台上使用 `default` 和命名的网络。

```afsim
// 示例：IADS 指挥官平台类型定义 comm
comm cmdr_net TEAM_DATALINK
    network_name <local:slave> // 作为下级接收命令
    internal_link data_mgr
    internal_link task_mgr
end_comm

// 示例：大型 SAM 营平台类型定义 comm
comm sub_net TEAM_DATALINK
    network_name <local:master> // 作为指挥官发送任务
    internal_link data_mgr
    internal_link task_mgr
end_comm
```

---

## 3. 报告命令 (`report_to`)

* **目的**：指定消息的去向——谁告诉谁，用于通过通信将航迹发送到指定的收件人。
* **使用要求**：脚本处理器和跟踪处理器需要使用此命令，但任务处理器不需要。

### 3.1 基于指挥链的报告
使用 `report_to` 命令，通过指定的通信设备将航迹报告给指挥链上的特定接收方。

* **接收方**：可以是 `commander`（指挥官）、`subordinates`（下级）或 `peers`（同级）。

```afsim
// 示例 1：向指挥官报告 (使用默认指挥链)
processor data_mgr WSF_TRACK_PROCESSOR
    report_to commander via blue_comm // 通过 blue_comm 报告给指挥官
end_processor

// 示例 2：向 STRIKERS 链的指挥官报告
processor data_mgr WSF_TRACK_PROCESSOR
    report_to command_chain STRIKERS commander via cmdr_net to sub_net
end_processor
```

### 3.2 不使用指挥链的报告 (显式链接)
可以直接通过平台名称、地址或组来指定接收方，无需依赖 `command_chain`。

```afsim
processor data_mgr WSF_TRACK_PROCESSOR
    // 显式链接到平台名称
    report_to platform 10_iads_cmdr comm blue_comm via blue_comm
    
    // 显式链接到 IP 地址 (更高级通信使用)
    report_to address 192.168.1.32/24 via blue_comm
    
    // 链接到小组 (Group)
    report_to_group blue via blue_comm
end_processor
```
