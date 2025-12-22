# 在主程序中的错误列表
当存在追踪有关的实体时，include应该先导入被追踪者的文件，再导入追踪者的文件

# 对于platform的错误列表
不能出现weapon的定义

# 对于platform_type的错误列表
**每一类platform_type定义只能出现一次**
不能出现speed
0.0 1700 kts是错误的，正确的语法是0.0 s 1700 kts，以此类推
radius 500 m probability 0.8 是错误的，正确的语法是radius_and_pk 500 m 0.8，以此类推
execute at time 600 sec absolute 是错误的，正确的语法是 execute at_time 600 s absolute，以此类推