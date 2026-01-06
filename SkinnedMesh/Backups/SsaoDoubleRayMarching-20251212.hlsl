//=============================================================================
// Ssao.hlsl by Frank Luna (C) 2015 All Rights Reserved.
// 修复：移除递归，用循环实现2次反射（解决崩溃问题）
//=============================================================================

cbuffer cbSsao : register(b0)
{
    float4x4 gProj;
    float4x4 gInvProj;
    float4x4 gProjTex;
    float4 gOffsetVectors[14];

    float4 gBlurWeights[3];
    float2 gInvRenderTargetSize;

    float gOcclusionRadius;
    float gOcclusionFadeStart;
    float gOcclusionFadeEnd;
    float gSurfaceEpsilon;
};

cbuffer cbRootConstants : register(b1)
{
    bool gHorizontalBlur;
};
 
Texture2D gNormalMap : register(t0);
Texture2D gDepthMap : register(t1);
Texture2D gAlbedoMap : register(t2);
Texture2D gRandomVecMap : register(t3);

SamplerState gsamPointClamp : register(s0);
SamplerState gsamLinearClamp : register(s1);
SamplerState gsamDepthMap : register(s2);
SamplerState gsamLinearWrap : register(s3);

static const int gSampleCount = 14;
static const int gMaxRaySteps = 6;
static const float gRayHitThreshold = 0.02f;
static const int gMaxReflectCount = 2; // 最多2次反射
static const float gReflectAtten = 0.5f; // 反射能量衰减
static const float gReflectDistScale = 0.8f; // 反射距离缩放（避免越界）
 
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
    vout.PosH = float4(2.0f * vout.TexC.x - 1.0f, 1.0f - 2.0f * vout.TexC.y, 0.0f, 1.0f);
    float4 ph = mul(vout.PosH, gInvProj);
    vout.PosV = ph.xyz / ph.w;
    return vout;
}

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

float NdcDepthToViewDepth(float z_ndc)
{
    float denom = z_ndc - gProj[2][2];
    denom = max(abs(denom), 0.0001f) * sign(denom);
    float viewZ = gProj[3][2] / denom;
    return viewZ;
}

// 【修复1：移除递归，改为单次步进函数（无嵌套调用）】
bool SingleRayStep(
    float3 rayStart,
    float3 rayDir,
    float maxDist,
    out float3 hitPos,
    out float2 hitTexC,
    out float3 hitNormal // 新增：输出命中点法线（用于二次反射）
)
{
    hitPos = float3(0.0f, 0.0f, 0.0f);
    hitTexC = float2(0.0f, 0.0f);
    hitNormal = float3(0.0f, 1.0f, 0.0f); // 默认法线

    float stepSize = maxDist / (float) gMaxRaySteps;
    float3 currPos = rayStart;

    for (int step = 0; step < gMaxRaySteps; step++)
    {
        currPos += rayDir * stepSize;
        if (length(currPos - rayStart) > maxDist)
            break;

        float4 projCurr = mul(float4(currPos, 1.0f), gProjTex);
        projCurr /= projCurr.w;
        projCurr.xy = saturate(projCurr.xy); // 强制限制在屏幕内

        float z_ndc = gDepthMap.SampleLevel(gsamDepthMap, projCurr.xy, 0.0f).r;
        float sceneZ = NdcDepthToViewDepth(z_ndc);

        if (abs(currPos.z) < 0.0001f)
            continue;

        float3 scenePos = (sceneZ / currPos.z) * currPos;

        if (length(currPos - scenePos) < gRayHitThreshold)
        {
            hitPos = scenePos;
            hitTexC = projCurr.xy;
            hitNormal = normalize(gNormalMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f).xyz);
            return true;
        }
    }
    return false;
}

// 【修复2：用循环实现多次反射（替代递归）】
float3 MultiReflectLight(
    float3 startPos,
    float3 startDir,
    float maxDist
)
{
    float3 totalLight = float3(0.0f, 0.0f, 0.0f);
    float3 currPos = startPos;
    float3 currDir = startDir;
    float currDist = maxDist;
    float atten = 1.0f; // 能量衰减累计

    // 循环实现最多2次反射（无递归）
    for (int reflectIdx = 0; reflectIdx < gMaxReflectCount; reflectIdx++)
    {
        float3 hitPos;
        float2 hitTexC;
        float3 hitNormal;

        if (SingleRayStep(currPos, currDir, currDist, hitPos, hitTexC, hitNormal))
        {
            // 采样当前反射的颜色，并叠加衰减
            float4 albedo = gAlbedoMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f);
            float dist = length(hitPos - currPos);
            float distAtten = pow(1.0f - saturate(dist / currDist), 2.0f);
            totalLight += albedo.rgb * distAtten * atten;

            // 更新下一次反射的参数
            currPos = hitPos;
            currDir = reflect(currDir, hitNormal); // 计算下一次反射方向
            currDist = maxDist * gReflectDistScale; // 缩小下一次反射距离
            atten *= gReflectAtten; // 衰减能量
        }
        else
        {
            break; // 未命中则终止反射
        }
    }
    return totalLight;
}
 
float4 PS(VertexOut pin) : SV_Target
{
    float3 n = gNormalMap.SampleLevel(gsamPointClamp, pin.TexC, 0.0f).xyz;
    float pz = gDepthMap.SampleLevel(gsamDepthMap, pin.TexC, 0.0f).r;
    pz = NdcDepthToViewDepth(pz);

    if (abs(pin.PosV.z) < 0.0001f)
        return float4(0.05f, 0.05f, 0.05f, 1.0f);

    float3 p = (pz / pin.PosV.z) * pin.PosV;
    float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * pin.TexC, 0.0f).rgb - 1.0f;

    float occlusionSum = 0.0f;
    float3 indirectLightSum = float3(0.0f, 0.0f, 0.0f);
    int validSampleCount = 0;

    for (int i = 0; i < gSampleCount; ++i)
    {
        float3 offset = reflect(gOffsetVectors[i].xyz, randVec);
        float flip = sign(dot(offset, n));
        float3 rayDir = normalize(flip * offset);

        // 【修复3：调用循环实现的多次反射函数（无递归）】
        float3 hitIndirectLight = MultiReflectLight(p, rayDir, gOcclusionRadius);

        // 首次命中的遮挡计算（沿用原始逻辑）
        float3 hitPos;
        float2 hitTexC;
        float3 hitNormal;
        bool isHit = SingleRayStep(p, rayDir, gOcclusionRadius, hitPos, hitTexC, hitNormal);

        if (isHit)
        {
            float distZ = p.z - hitPos.z;
            float dp = max(dot(n, normalize(hitPos - p)), 0.0f);
            occlusionSum += dp * OcclusionFunction(distZ);

            if (distZ > -0.1f && dp > 0.01f)
            {
                indirectLightSum += hitIndirectLight;
                validSampleCount++;
            }
        }
        else
        {
            // 未命中时沿用原始逻辑
            float3 q = p + flip * gOcclusionRadius * offset;
            float4 projQ = mul(float4(q, 1.0f), gProjTex);
            projQ /= projQ.w;
            projQ.xy = saturate(projQ.xy);

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

    occlusionSum = validSampleCount > 0 ? occlusionSum / validSampleCount : 0.0f;
    indirectLightSum = validSampleCount > 0 ? indirectLightSum / validSampleCount : float3(0.0f, 0.0f, 0.0f);

    float access = 1.0f - occlusionSum;
    float occlusion = saturate(pow(access, 2.0f));
    float3 ssgiColor = saturate(indirectLightSum + float3(0.05f, 0.05f, 0.05f));
	
    return float4(ssgiColor, occlusion);
}