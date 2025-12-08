# AFSIM 脚本语言与脚本处理器 (Script Language and Processors)

## 1. 概述与基本语法

* **脚本语言**：AFSIM 提供自定义脚本语言，用户可以基于仿真事件执行复杂指令。
* **语法风格**：类似于 C# 和 Java，需要基本的编程技能。
* **基本语法规则**：
    * 没有严格的缩进、换行等规则。
    * 行以分号 (`;`) 结尾。
    * 变量在使用之前必须声明。
    * 注释行以符号 `#` 或 C++ 样式 (`//`, `/*..*/`) 开始。
* **作用域**：脚本变量/方法可以定义在全局、平台、处理器或行为级别。

### 1.1 基本类型和运算符

* **基本类型**：
    * `int` (32位整数)
    * `bool` (布尔值: `true` 或 `false`)
    * `double` (双精度浮点值)
    * `string` (字符串)
* **复杂类型 (静态)**：_BUILTIN_, Calendar, Earth, Math, Moon, Sun, System。
* **复杂类型 (可变/容器)**：Array<T>, Map<T1, T2>, Set<T>, Vec3, Struct。

### 1.2 类型转换示例
```afsim
int iValue = 42;
iValue = iValue + 4;
string sValue = (string) iValue; // sValue now contains the string "46"
iValue = (int) sValue;          // iValue now contains the integer 46
```

* `MATH.PI()`: 返回 Pi 的值。
* `MATH.M_PER_FT()`: 返回每英尺的米数。
* `MATH.Pow(double aX, double aY)`: 返回 X 的 Y 次幂。
* `MATH.Floor(double aNumber)`: 返回不大于给定数字的最大整数。

---

## 2. 脚本块与结构

### 2.1 脚本方法 (`script` block)
用于定义可在其他脚本中调用的对象类型或函数。

```afsim
script bool IsScriptingFun()
    return true;
end script

script void RecordPlatformName (string aFileName, WsfPlatform aPlatform)
    // FileIO 对象必须在 script_variables 中声明
    gout.Open(aFileName, "append");
    gout.Writeln(aPlatform.Name());
    gout.Close();
end script
```

### 2.2 脚本结构 (`script_struct`)
用于创建自定义的复杂数据类型，类似于面向对象编程中的类。

```afsim
script_struct Car
    script_variables
        WsfGeoPoint position = {};
        Vec3 velocity = {};
        string color = "red";
    end_script_variables
    
    script void Honk()
        writeln(color, " car honks");
    end_script
end_script_struct

// 实例化和使用
on_initialize
    // 创建一个新的 struct 实例
    Car car = Car();
    car.color = "blue";
    car.Honk(); // 输出: blue car honks
    writeln(car);
end_on_initialize
```

### 2.3 箭头运算符 (`->`)
用于访问平台 (Platform) 等复杂对象中的用户变量（即在 `script_variables` 中定义的变量）。

```afsim
// 平台 p 中定义了 double x = 1.75;
// 在另一个平台的 execute 块中访问：
execute at time 1 s absolute
    WsfPlatform p = FindPlatform("p");
    writeln(p->x); // 输出: 1.75
end execute
```

### 2.4 条件与循环
支持标准的 `if/else if/else`, `while/do while`, `for`, 和 `foreach` 循环。

```afsim
foreach (string aKey: double aData in myMap) // 访问 Map 的键和值
{
    print("key, data", aKey, ", ", aData);
    if (aData > 2.0) break;
}
```

---

## 3. 脚本处理器 (`WSF_SCRIPT_PROCESSOR`)

### 3.1 通用脚本界面要素 (Universal Script Interface)
AFSIM 组件（如仿真、平台、处理器、武器）实现了通用脚本接口，包含：
* 脚本变量块 (`script_variables`)
* 脚本方法块 (`script` blocks)
* 脚本控制命令 (`execute`, `on_initialize`, `on_update`, `on_message`)

### 3.2 预定义变量 (上下文相关)
* `TIME_NOW`: 模拟中的当前时间。
* `PLATFORM`: 当前平台对象（在平台或其部件上下文中）。
* `PROCESSOR`: 当前动作处理器对象。
* `TRACK`: 状态机中正在处理的当前跟踪。
* `MasterTrackList()`: 平台维护的航迹列表。

### 3.3 处理器脚本生命周期
* `on_initialize`: 平台初始化时的第一阶段。
* `on_initialize2`: 平台部件初始化完成后的第二阶段（通常用于确保所有部件和航迹已设置）。
* `on_update`: 在 `update_interval` 设定的时间间隔内重复执行。
* `on_message`: 收到消息时执行。

### 3.4 示例：开火脚本处理器 (`BOMBER_WEAPON_RELEASE`)
脚本处理器 (`WSF_SCRIPT_PROCESSOR`) 可实现复杂战术逻辑。

**A. 脚本变量 (`script_variables`):**
```afsim
processor BOMBER_WEAPON_RELEASE WSF_SCRIPT_PROCESSOR
    script_variables
        string wpn39 = "gbu-39";     // 武器名称
        string LARmeters = "lar_meters"; // 武器射程的 AuxData 键名
        Array<bool> tgt_engaged = Array<bool>(); // 记录目标是否已被攻击
    end_script_variables
    // ...
end_processor
```

**B. 开火函数 (`fireWeapon`):**
该函数封装了复杂的开火逻辑，包括射程检查、武器剩余数量检查和高度检查。

```afsim
script bool fireWeapon (WsfTrack tTrack, string twpn)
    bool shot = false;
    WsfWeapon tempWeapon = PLATFORM.Weapon(tWpn);
    
    if (tempWeapon.IsValid())
    {
        // 检查武器的射程 AuxData
        double tempRange = tempWeapon.AuxDataDouble(LARmeters);
        
        if (PLATFORM.GroundRangeTo(tTrack) < tempRange          // # 在射程内
            && tempWeapon.QuantityRemaining() > 0               // # 武器数量充足
            && PLATFORM.Altitude() >= 7620)                     // # 高度高于 25,000 ft
        {
            shot = tempWeapon.FireSalvo(tTrack, 2); // 发射 2 枚
        }
        
        if (shot)
        {
            // 发射成功后打印输出
            writeln(PLATFORM.Name(), " fired ", twpn, " at ", tTrack.TargetName(), " at time ", TIME_NOW);
        }
    }
    return shot;
end script
```

**C. 周期更新 (`on_update`):**
处理器的主逻辑，定期检查所有目标并尝试开火。

```afsim
update_interval 3.0 s
on_update
    for (int i=0; i < PLATFORM.MasterTrackList().Count(); i+=1)
    {
        if (!tgt_engaged[i]) // # 确认目标尚未参与交战
        {
            WsfTrack tempTrack = PLATFORM.MasterTrackList().TrackEntry(i);
            
            // 优先检查 GBU-39
            if (fireWeapon(tempTrack, wpn39))
            {
                tgt_engaged[i] = true;
            }
            // 接着检查 GBU-38
            else if (fireWeapon(tempTrack, wpn38))
            {
                tgt_engaged[i] = true;
            }
        }
    }
end_on_update
end_processor
```

### 3.5 连接处理器到平台 (Final Integration)
要在平台上启用这个逻辑，必须在平台类型定义中实例化该处理器。

```afsim
platform_type BOMBER WSF_PLATFORM
    // ... 武器定义 ...
    
    // 实例化处理器
    processor fire-em BOMBER_WEAPON_RELEASE
    end_processor
    
    // ...
end_platform_type
```
```