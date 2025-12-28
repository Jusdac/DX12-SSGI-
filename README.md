# 基于DX12龙书上SSAO开发的SSGI方案
实现了光线步进的光线追踪算法
新增小功能：固定可自定义小图标
<img width="796" height="634" alt="image" src="https://github.com/user-attachments/assets/820c6a8e-4bdb-4310-b77e-10ec4544a441" />
右下角上方是间接光debug图下面是遮蔽度debug图

<img width="805" height="632" alt="image" src="https://github.com/user-attachments/assets/43eec341-e9fa-438d-84c2-c011b05f7175" />
可以看到柱子下方比较明显的绿色光晕，即为其反射到地面上的间接光。
修改参数可以在SkinnedMesh\Shaders\Ssao.hlsl中修改：
// 核心常量（集中管理）
static const int gSampleCount = 14;
static const int gMaxRaySteps = 12;
static const float gRayHitThreshold = 0.6f;
static const int gMaxReflectCount = 2;
static const float gReflectAtten = 0.9f;
static const float gReflectDistScale = 0.9f;
static const float gMinThreshold = 0.0001f;
static const float3 gMinAlbedo = float3(0.1f, 0.1f, 0.1f);
 
