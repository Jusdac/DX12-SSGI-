
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

// 核心常量（集中管理）
static const int gSampleCount = 14;
static const int gMaxRaySteps = 12;
static const float gRayHitThreshold = 0.6f;
static const int gMaxReflectCount = 2;
static const float gReflectAtten = 0.9f;
static const float gReflectDistScale = 0.9f;
static const float gMinThreshold = 0.0001f;
static const float3 gMinAlbedo = float3(0.1f, 0.1f, 0.1f);
 
static const float2 gTexCoords[6] =
{
    float2(0.0f, 1.0f), float2(0.0f, 0.0f), float2(1.0f, 0.0f),
    float2(0.0f, 1.0f), float2(1.0f, 0.0f), float2(1.0f, 1.0f)
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

// 遮挡计算（精简版）
float OcclusionFunction(float distZ)
{
    if (distZ <= gSurfaceEpsilon)
        return 0.0f;
    float fadeLen = max(gOcclusionFadeEnd - gOcclusionFadeStart, gMinThreshold);
    return saturate((gOcclusionFadeEnd - distZ) / fadeLen);
}

// NDC转视图深度（安全除法）
float NdcDepthToViewDepth(float z_ndc)
{
    float denom = z_ndc - gProj[2][2];
    denom = (abs(denom) < gMinThreshold) ? gMinThreshold * sign(denom) : denom;
    return gProj[3][2] / denom;
}

// 单次射线检测（精简逻辑）
bool SingleRayStep(
    float3 rayStart, float3 rayDir, float maxDist,
    out float3 hitPos, out float2 hitTexC, out float3 hitNormal
)
{
    hitPos = float3(0, 0, 0);
    hitTexC = float2(0, 0);
    hitNormal = float3(0, 1, 0);
    if (maxDist < gMinThreshold || length(rayDir) < gMinThreshold)
        return false;

    float stepSize = maxDist / gMaxRaySteps;
    float3 currPos = rayStart;
    float3 rayDirNorm = normalize(rayDir);
    // 射线起点抬离表面
    float bias = 0.02f; // 偏移量
    currPos = rayStart + rayDirNorm * bias; // 起点沿射线方向偏移，跳过自身

    for (int step = 0; step < gMaxRaySteps; step++)
    {
        currPos += rayDirNorm * stepSize;
        if (distance(currPos, rayStart) > maxDist)
            break;

        float4 projCurr = mul(float4(currPos, 1.0f), gProjTex);
        projCurr.xyz /= projCurr.w;
        projCurr.xy = saturate(projCurr.xy);

        float sceneZ = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, projCurr.xy, 0.0f).r);
        if (abs(currPos.z) < gMinThreshold)
            continue;

        float3 scenePos = currPos * (sceneZ / currPos.z);
        if (distance(currPos, scenePos) < gRayHitThreshold && sceneZ < rayStart.z)
        {
            hitPos = scenePos;
            hitTexC = projCurr.xy;
            hitNormal = normalize(gNormalMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f).xyz);
            hitNormal.z = -hitNormal.z; // 法线Z轴修正
            return true;
        }
    }
    return false;
}

// 多反射计算（精简循环）
float3 MultiReflectLight(
    float3 startPos, float3 startDir, float maxDist,
    out bool firstHit, out float3 firstHitPos, out float2 firstHitTexC, out float3 firstHitNormal
)
{
    // 初始化输出参数（默认未命中）
    firstHit = false;
    firstHitPos = float3(0, 0, 0);
    firstHitTexC = float2(0, 0);
    firstHitNormal = float3(0, 1, 0);

    float3 totalLight = float3(0, 0, 0);
    float3 currPos = startPos;
    float3 currDir = normalize(startDir);
    float currDist = maxDist;
    float atten = 1.0f;

    // 循环实现多次反射
    for (int reflectIdx = 0; reflectIdx < gMaxReflectCount; reflectIdx++)
    {
        float3 hitPos;
        float2 hitTexC;
        float3 hitNormal;

        // 仅执行一次射线检测（复用给SSAO）
        bool hit = SingleRayStep(currPos, currDir, currDist, hitPos, hitTexC, hitNormal);
        
        // 首次反射的结果，输出给SSAO使用
        if (reflectIdx == 0)
        {
            firstHit = hit;
            firstHitPos = hitPos;
            firstHitTexC = hitTexC;
            firstHitNormal = hitNormal;
        }

        if (!hit)
            break; // 未命中则终止反射

        // 采样反照率 + 距离衰减
        float3 albedo = max(gAlbedoMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f).rgb, gMinAlbedo);
        
        float distAtten = pow(1.0f - saturate(distance(hitPos, currPos) / currDist), 1.0f) * 3.0f;
        totalLight += albedo * distAtten * atten;

        // 更新反射参数
        currPos = hitPos;
        currDir = reflect(currDir, hitNormal);
        currDist *= gReflectDistScale;
        atten *= gReflectAtten;
    }
    return totalLight;
}
 

float4 PS(VertexOut pin) : SV_Target
{
    // 1. 基础数据采样 & 有效性判断
    float2 texC = pin.TexC;
    float3 normal = normalize(gNormalMap.SampleLevel(gsamPointClamp, texC, 0.0f).xyz);
    float viewZ = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, texC, 0.0f).r);

    if (abs(pin.PosV.z) < gMinThreshold || abs(viewZ) < gMinThreshold)
        return float4(0.05f, 0.05f, 0.05f, 1.0f);

    float3 posV = (viewZ / pin.PosV.z) * pin.PosV;
    float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * texC, 0.0f).rgb - 1.0f;

    // 2. 初始化累计变量
    float occlusionSum = 0.0f;
    float3 indirectLightSum = float3(0, 0, 0);
    int validSampleCount = 0;

    // 3. 采样循环（核心逻辑：无分支 + 少变量）
    for (int i = 0; i < gSampleCount; ++i)
    {
      // 1. 计算射线方向
        float3 offset = reflect(gOffsetVectors[i].xyz, randVec);
        float flip = sign(dot(offset, normal));
        float3 rayDir = normalize(flip * offset);
        //防止ray和法线重合
        rayDir = normalize(rayDir + 0.1f * normal); // 偏向法线方向，避免向内
        float cosAngle = dot(rayDir, normal);
        if (cosAngle < 0.17f) // cos(80°)≈0.17，小于10度则舍弃
            continue;
        
    // 2. 仅调用一次：同时拿到「首次命中（SSAO）」和「反射光（SSGI）」
        float3 hitLight;
        bool isHit;
        float3 hitPos, hitNormal;
        float2 hitTexC;
        hitLight = MultiReflectLight(
        posV, rayDir, gOcclusionRadius,
        isHit, hitPos, hitTexC, hitNormal);

        // 3.3 预计算：未命中分支的基础数据（作为默认值）
        float3 q = posV + flip * gOcclusionRadius * offset;
        float4 projQ = mul(float4(q, 1.0f), gProjTex);
        projQ.xyz /= projQ.w;
        projQ.xy = saturate(projQ.xy);

        float rz = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r);
        float nonHitValid = step(gMinThreshold, abs(q.z)); // 未命中有效标志（0/1）

        // 3.4 定义核心变量（默认=未命中值，命中时覆盖）
        float3 samplePos = q * (rz / q.z);
        float distZ = posV.z - rz;
        float dp = max(dot(normal, normalize(samplePos - posV)), 0.0f);
        float2 sampleTexC = projQ.xy;

        // 3.5 未命中时的间接光（默认值）
        float spatialDist = distance(samplePos, posV);
        float distAtten = pow(1.0f - saturate(spatialDist / gOcclusionRadius), 0.8f);
        hitLight = gAlbedoMap.SampleLevel(gsamPointClamp, sampleTexC, 0.0f).rgb * distAtten;

        // 3.6 命中时覆盖核心变量（单掩码+批量赋值，减少重复计算）
        float hitMask = isHit ? 1.0f : 0.0f;
        if (isHit)
        {
            samplePos = hitPos;
            distZ = posV.z - hitPos.z;
            dp = max(dot(normal, normalize(hitPos - posV)), 0.0f);
            sampleTexC = hitTexC;
        }

        // 3.7 有效性过滤（替代continue + 无效样本置0）
        float validMask = (isHit ? 1.0f : nonHitValid);
        distZ *= validMask;
        dp *= validMask;
        hitLight *= validMask;

        // 3.8 累计结果
        occlusionSum += dp * OcclusionFunction(distZ);
        float indirectValid = step(-0.1f, distZ) * step(0.01f, dp);
        // 过滤近距离采样 距离<0.05视为自身
        float minSampleDist = step(0.05f, distance(samplePos, posV));
        indirectValid *= minSampleDist;
        indirectLightSum += hitLight * indirectValid * 2.0f;
        validSampleCount += int(indirectValid * validMask);
    }

    // 4. 归一化 + 最终结果计算
    float validFactor = validSampleCount > 0 ? 1.0f / validSampleCount : 0.0f;
    occlusionSum *= validFactor;
    indirectLightSum *= validFactor;

    float occlusion = saturate(pow(1.0f - occlusionSum, 2.0f));
    float3 ssgiColor = saturate(indirectLightSum + float3(0.15f, 0.15f, 0.15f));

    return float4(ssgiColor, occlusion);
}