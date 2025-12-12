
#include "Common.hlsl"
struct VertexIn
{
    float3 PosL : POSITION;
    float2 TexC : TEXCOORD;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float2 TexC : TEXCOORD;
};
// 常量：矩形大小缩放因子（控制图标最终显示大小，可按需调整）
#define ICON_SCALE 0.05f // 0.05f 对应屏幕的 5% 大小，值越大图标越大
#define OFFSET 1.0f
VertexOut VS(VertexIn vin)
{
    VertexOut vout = (VertexOut) 0.0f;
    // Already in homogeneous clip space.
    vout.PosH = float4(vin.PosL, 1.0f);
    if (vin.PosL.x == 0 && vin.PosL.y == 0 )
    {
        //vout.PosH = float4(-0.5f,0.0f,0.0f, 1.0f);
        vout.PosH = IconPos + float4(-OFFSET, OFFSET, 0.0f, 0.0f);
        vout.PosH.z = 0;

    }
    if (vin.PosL.x == 0 && vin.PosL.y == -1)
    {
        //vout.PosH = float4(-0.5f, -0.5f, 0.0f, 1.0f);
        vout.PosH = IconPos + float4(-OFFSET, -OFFSET, 0.0f, 0.0f);
        vout.PosH.z = 0;
    }
    if (vin.PosL.x == 1 && vin.PosL.y == 0)
    {
        //vout.PosH = float4(0.5f, 0.5f, 0.0f, 1.0f);
        vout.PosH = IconPos + float4(OFFSET, OFFSET, 0.0f, 0.0f);
        vout.PosH.z = 0;
    }
    if (vin.PosL.x == 1 && vin.PosL.y == -1)  
    {
        //vout.PosH = float4(0.5f, -0.8f, 0.0f, 1.0f);
        vout.PosH = IconPos + float4(OFFSET, -OFFSET, 0.0f, 0.0f);
        vout.PosH.z = 0;
    }
    vout.TexC = vin.TexC;
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
   // float4 Color = float4(gTextureMaps[6].Sample(gsamLinearWrap, pin.TexC).rrr, 1.0f);
    float4 Color = (gTextureMaps[6].Sample(gsamLinearWrap, pin.TexC));
   // return float4(gTextureMaps[6].Sample(gsamLinearWrap, pin.TexC).rrr, 1.0f);
    clip(Color.a - 0.1f);
    return Color;
}

