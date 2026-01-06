# 基于DX12龙书上SSAO开发的SSGI方案
<img width="792" height="629" alt="image" src="https://github.com/user-attachments/assets/036c994f-fb6f-43c0-abf3-f64b501aeb9c" />
右下角上方是间接光debug图下面是遮蔽度debug图

可以看到柱子下方比较明显的绿色光晕，即为其反射到地面上的间接光。

V1.2更新：1-6,2026

1:优化了反射逻辑，消减了冗余的计算

2：支持了Hi-z剔除算法，提升了性能

3：支持了sky handling，间接光可以采样天空盒
 
