//=============================================================================
// Ssao.hlsl by Frank Luna (C) 2015 All Rights Reserved.
// 新增：极简光线步进（仅核心逻辑，无复杂物理计算）
//=============================================================================

cbuffer cbSsao : register(b0)
{
    float4x4 gProj;
    float4x4 gInvProj;
    float4x4 gProjTex;
    float4 gOffsetVectors[14];

    // For SsaoBlur.hlsl
    float4 gBlurWeights[3];

    float2 gInvRenderTargetSize;

    // Coordinates given in view space.
    float gOcclusionRadius;
    float gOcclusionFadeStart;
    float gOcclusionFadeEnd;
    float gSurfaceEpsilon;
};

cbuffer cbRootConstants : register(b1)
{
    bool gHorizontalBlur;
};
 
// Nonnumeric values cannot be added to a cbuffer.
Texture2D gNormalMap : register(t0);
Texture2D gDepthMap : register(t1);
Texture2D gAlbedoMap : register(t2);
Texture2D gRandomVecMap : register(t3);

SamplerState gsamPointClamp : register(s0);
SamplerState gsamLinearClamp : register(s1);
SamplerState gsamDepthMap : register(s2);
SamplerState gsamLinearWrap : register(s3);

static const int gSampleCount = 14;
// 光线步进基础常量（极简版）
static const int gMaxRaySteps = 6; // 少量步进保证性能
static const float gRayHitThreshold = 0.02f; // 宽松判定阈值
 
static const float2 gTexCoords[6] =
{
    float2(0.0f, 1.0f),
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 0.0f),
    float2(1.0f, 1.0f)
};
 
struct VertexOut
{
    float4 PosH : SV_POSITION;
    float3 PosV : POSITION;
    float2 TexC : TEXCOORD0;
};

VertexOut VS(uint vid : SV_VertexID)
{
    VertexOut vout;

    vout.TexC = gTexCoords[vid];

    // Quad covering screen in NDC space.
    vout.PosH = float4(2.0f * vout.TexC.x - 1.0f, 1.0f - 2.0f * vout.TexC.y, 0.0f, 1.0f);
 
    // Transform quad corners to view space near plane.
    float4 ph = mul(vout.PosH, gInvProj);
    vout.PosV = ph.xyz / ph.w;

    return vout;
}

// 原始OcclusionFunction（保留不变）
float OcclusionFunction(float distZ)
{
    float occlusion = 0.0f;
    if (distZ > gSurfaceEpsilon)
    {
        float fadeLength = gOcclusionFadeEnd - gOcclusionFadeStart;
        occlusion = saturate((gOcclusionFadeEnd - distZ) / fadeLength);
    }
    return occlusion;
}

// 修复除零风险的深度转换（保证编译）
float NdcDepthToViewDepth(float z_ndc)
{
    float denom = z_ndc - gProj[2][2];
    denom = max(abs(denom), 0.0001f) * sign(denom); // 防止除零
    float viewZ = gProj[3][2] / denom;
    return viewZ;
}

// 极简光线步进函数（核心逻辑，无复杂计算）
bool RayMarchSimple(
    float3 rayStart, // 射线起点（视图空间）
    float3 rayDir, // 射线方向（归一化）
    float maxDist, // 最大步进距离
    out float3 hitPos, // 相交点（视图空间）
    out float2 hitTexC // 相交点纹理坐标
)
{
    // 初始化输出
    hitPos = float3(0.0f, 0.0f, 0.0f);
    hitTexC = float2(0.0f, 0.0f);

    // 步进长度 = 最大距离 / 总步数
    float stepSize = maxDist / (float) gMaxRaySteps;
    float3 currPos = rayStart;

    // 循环步进（无unroll，简化编译）
    for (int step = 0; step < gMaxRaySteps; step++)
    {
        // 沿射线推进一步
        currPos += rayDir * stepSize;

        // 超出最大距离则终止
        if (length(currPos - rayStart) > maxDist)
            break;

        // 投影到NDC空间，生成纹理坐标
        float4 projCurr = mul(float4(currPos, 1.0f), gProjTex);
        projCurr /= projCurr.w;
        projCurr.xy = saturate(projCurr.xy); // 限制在屏幕内

        // 采样当前位置的深度
        float z_ndc = gDepthMap.SampleLevel(gsamDepthMap, projCurr.xy, 0.0f).r;
        float sceneZ = NdcDepthToViewDepth(z_ndc);

        // 防止除以零
        if (abs(currPos.z) < 0.0001f)
            continue;

        // 重构场景真实位置
        float3 scenePos = (sceneZ / currPos.z) * currPos;

        // 极简相交判定：距离小于阈值即认为命中
        if (length(currPos - scenePos) < gRayHitThreshold)
        {
            hitPos = scenePos;
            hitTexC = projCurr.xy;
            return true;
        }
    }

    return false; // 未命中
}
 
float4 PS(VertexOut pin) : SV_Target
{
	// 1. 重构当前像素视图空间信息（保留原始逻辑）
    float3 n = gNormalMap.SampleLevel(gsamPointClamp, pin.TexC, 0.0f).xyz;
    float pz = gDepthMap.SampleLevel(gsamDepthMap, pin.TexC, 0.0f).r;
    pz = NdcDepthToViewDepth(pz);

    // 防止除零
    if (abs(pin.PosV.z) < 0.0001f)
        return float4(0.05f, 0.05f, 0.05f, 1.0f);

    float3 p = (pz / pin.PosV.z) * pin.PosV;
    float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * pin.TexC, 0.0f).rgb - 1.0f;

    float occlusionSum = 0.0f;
    float3 indirectLightSum = float3(0.0f, 0.0f, 0.0f);
    int validSampleCount = 0;

	// 2. 采样循环（保留原始结构，新增光线步进）
    for (int i = 0; i < gSampleCount; ++i)
    {
        float3 offset = reflect(gOffsetVectors[i].xyz, randVec);
        float flip = sign(dot(offset, n));

        // 生成射线方向（基于原始offset，极简处理）
        float3 rayDir = normalize(flip * offset);

        // 光线步进检测相交
        float3 hitPos;
        float2 hitTexC;
        bool isHit = RayMarchSimple(p, rayDir, gOcclusionRadius, hitPos, hitTexC);

        if (isHit)
        {
            // 命中时使用光线步进结果计算遮挡
            float distZ = p.z - hitPos.z;
            float dp = max(dot(n, normalize(hitPos - p)), 0.0f);
            occlusionSum += dp * OcclusionFunction(distZ);

            // 极简SSGI间接光计算（无物理公式）
            if (distZ > -0.1f && dp > 0.01f)
            {
                float4 rAlbedo = gAlbedoMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f);
                float spatialDistance = length(hitPos - p);
                float distanceAtten = pow(1.0f - saturate(spatialDistance / gOcclusionRadius), 2.0f);
                
                indirectLightSum += rAlbedo.rgb * distanceAtten;
                validSampleCount++;
            }
        }
        else
        {
            // 未命中时沿用原始逻辑（保证兼容）
            float3 q = p + flip * gOcclusionRadius * offset;
            float4 projQ = mul(float4(q, 1.0f), gProjTex);
            projQ /= projQ.w;

            float rz = gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r;
            rz = NdcDepthToViewDepth(rz);

            if (abs(q.z) < 0.0001f)
                continue;

            float3 r = (rz / q.z) * q;
            float distZ = p.z - r.z;
            float dp = max(dot(n, normalize(r - p)), 0.0f);
            occlusionSum += dp * OcclusionFunction(distZ);

            if (distZ > -0.1f && dp > 0.01f)
            {
                float4 rAlbedo = gAlbedoMap.SampleLevel(gsamPointClamp, projQ.xy, 0.0f);
                float spatialDistance = length(r - p);
                float distanceAtten = pow(1.0f - saturate(spatialDistance / gOcclusionRadius), 2.0f);
                
                indirectLightSum += rAlbedo.rgb * distanceAtten;
                validSampleCount++;
            }
        }
    }

	// 3. 归一化（防止除以零）
    occlusionSum = validSampleCount > 0 ? occlusionSum / validSampleCount : 0.0f;
    indirectLightSum = validSampleCount > 0 ? indirectLightSum / validSampleCount : float3(0.0f, 0.0f, 0.0f);

    float access = 1.0f - occlusionSum;
    float occlusion = saturate(pow(access, 2.0f));
    float3 ssgiColor = saturate(indirectLightSum + float3(0.05f, 0.05f, 0.05f));
	
    return float4(ssgiColor, occlusion);
}