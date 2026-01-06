//===========================================================
//ssao.hlsl(SSGI) by ZSD on the base of Frank Luna
//===========================================================
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
    float4x4 gInvView;
};

cbuffer cbRootConstants : register(b1)
{
    bool gHorizontalBlur;
};
 
Texture2D gNormalMap : register(t0);
Texture2D gDepthMap : register(t1);
Texture2D gAlbedoMap : register(t2);
Texture2D gHizMap : register(t3);
Texture2D gRandomVecMap : register(t4);
TextureCube gCubeMap : register(t5);

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

// 单次射线检测
bool SingleRayStep(
    float3 rayStart,      //射线起点
    float3 rayDir,        //射线方向
    float maxDist,        //最大探测距离
    out float3 hitPos,    //输出命中点的世界位置
    out float2 hitTexC,   //命中点的屏幕uv
    out float3 hitNormal  //命中点的法线
)
{  //初始化
    hitPos = float3(0, 0, 0);
    hitTexC = float2(0, 0);
    hitNormal = float3(0, 1, 0);
    
    //检查：如果射线长度太短或者方向几乎为0则直接舍弃
    if (maxDist < gMinThreshold || length(rayDir) < gMinThreshold)
        return false;

    
    float stepSize = maxDist / gMaxRaySteps;  //每一步的距离
    float3 currPos = rayStart;                //当前步进位置
    float3 rayDirNorm = normalize(rayDir);    //单位化方向向量，确保步进均匀
    
    // 射线起点抬离表面，防止反射光撞到自己！
    float bias = 0.02f; // 偏移量
    currPos = rayStart + rayDirNorm * bias; // 起点沿射线方向偏移，跳过自身

    for (int step = 0; step < gMaxRaySteps; step++)  //RayMarching
    {
        currPos += rayDirNorm * stepSize;  //每次朝前走一步
        if (distance(currPos, rayStart) > maxDist)  //安全边界：防止超出范围
            break;

        //  Hi-Z优化！！！
        float4 projCurr = mul(float4(currPos, 1.0f), gProjTex);
        float2 uv = saturate(projCurr.xy / projCurr.w);
        // 采样 Hi-Z（假设存储的是 view-space 深度绝对值）
        float hizDepth = gHizMap.SampleLevel(gsamLinearWrap, uv, 0.0f).r;
        // 当前 ray 的 view-space 深度（取正值）
        float rayDepth = -currPos.z; // view space z is negative
        // 如果 ray 已经比 Hi-Z 记录的最近遮挡还要深很多 → 不可能再命中，提前退出
        if (rayDepth > hizDepth + gOcclusionRadius)
            break;

        //将当前的3D位置投影回屏幕UV（复用 projCurr）
        projCurr.xy = uv; // reuse computed uv

        //采样深度图，获取该uv处的场景深度
        float sceneZ = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, uv, 0.0f).r);
        if (abs(currPos.z) < gMinThreshold) //如果当前点z接近0，靠近近平面，跳过！
            continue;
        
        //重构该uv处的3D场景位置！
        float3 scenePos = currPos * (sceneZ / currPos.z);
        if (distance(currPos, scenePos) < gRayHitThreshold && sceneZ > rayStart.z)//判断命中
        {
            //判断空间上是否接近 && 深度更大
            hitPos = scenePos;   //保存命中信息
            hitTexC = projCurr.xy;   //起始点的uv
            hitNormal = normalize(gNormalMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f).xyz);
            hitNormal.z = -hitNormal.z; // 法线Z轴修正
            return true;
        }
    }
    return false;
}

// 多反射计算
float3 MultiReflectLight(
   float3 startPos, // 起始位置（view space）
    float3 startDir, // 初始方向（view space）
    float3 skyDirW,
    float maxDist, // 最大初始探测距离
    out bool firstHit, // 输出：第一次是否命中
    out float3 firstHitPos, // 第一次命中位置
    out float2 firstHitTexC, // 第一次命中 UV
    out float3 firstHitNormal // 第一次命中法线
)
{
    // 初始化输出参数（默认未命中）
    firstHit = false;
    firstHitPos = float3(0, 0, 0);
    firstHitTexC = float2(0, 0);
    firstHitNormal = float3(0, 1, 0);

    float3 totalLight = float3(0, 0, 0); //累计间接光
    float3 currPos = startPos; //当前光线起点
    float3 currDir = normalize(startDir); //方向
    float currDist = maxDist; //最大探测距离
    float atten = 1.0f; //光照衰减因子，会随着每次反射逐渐减小

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
        if (hit)
        {
            // 命中：采样表面反照率 + 距离衰减
            float3 albedo = max(gAlbedoMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f).rgb, gMinAlbedo);
            float distAtten = pow(1.0f - saturate(distance(hitPos, currPos) / currDist), 1.0f) * 3.0f;
            totalLight += albedo * distAtten * atten;

            // 更新反射
            currPos = hitPos;
            currDir = reflect(currDir, hitNormal);
            currDist *= gReflectDistScale;
            atten *= gReflectAtten;
        }
        else
        {
            // ====== 【Correct Sky Handling】======
            float3 skyColor = gCubeMap.Sample(gsamLinearWrap, skyDirW).rgb;
            // 限制天空盒颜色贡献，避免过亮
            totalLight += skyColor * atten * 0.2f; // 尝试不同的系数，如 0.2f
            break;
             // ===================================
        }
    }
    return totalLight;
}
 

float4 PS(VertexOut pin) : SV_Target
{
    // 1. 基础数据采样 & 有效性判断
    float2 texC = pin.TexC;
    float3 normal = normalize(gNormalMap.SampleLevel(gsamPointClamp, texC, 0.0f).xyz);
    float viewZ = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, texC, 0.0f).r);

    //安全检查：剔除靠近相机平面（z ≈ 0）的无效像素，若任一深度接近 0，说明该像素可能属于 UI、天空盒或无效区域。
    if (abs(pin.PosV.z) < gMinThreshold || abs(viewZ) < gMinThreshold)
        return float4(0.05f, 0.05f, 0.05f, 1.0f); //返回微弱灰 (0.05) 作为 fallback，避免黑屏

    //重建 view-space 3D 位置
    float3 posV = (viewZ / pin.PosV.z) * pin.PosV; //重建位置
    //采样低频随机向量，用于扰动采样方向以减少噪点。
    float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * texC, 0.0f).rgb - 1.0f;
    // 2. 初始化累计变量
    float occlusionSum = 0.0f;
    float3 indirectLightSum = float3(0, 0, 0);
    int validSampleCount = 0;

    // 3. 采样循环
    for (int i = 0; i < gSampleCount; ++i)
    {
      // 1. 计算射线方向
        float3 offset = reflect(gOffsetVectors[i].xyz, randVec); //gOffsetVectors[i] 以 randVec 为法线做镜面反射
        float flip = sign(dot(offset, normal)); //计算 offset 与表面法线的点积符号,sign(x) 返回 -1, 0, 或 1。
        float3 rayDir = normalize(flip * offset); //若 offset 在背面（dot < 0），则反向（flip = -1）。
        //防止ray和法线重合
        rayDir = normalize(rayDir + 0.1f * normal); // 偏向法线方向，避免向内
        float cosAngle = dot(rayDir, normal); //计算射线方向与法线的夹角余弦。
        //cosAngle = 1 → 完全垂直；cosAngle = 0 → 完全平行。
        if (cosAngle < 0.17f) // cos(80°)≈0.17，小于10度则舍弃,只保留与法线夹角 小于 80° 的射线（即仰角 > 10°）
            continue;
        
       // 2. 仅调用一次：同时拿到首次命中（SSAO 和反射光（SSGI
        float3 hitLight; //累计间接光照（RGB）
        bool isHit; //第一次是否命中（用于 SSAO）
        float3 hitPos, hitNormal;
        float2 hitTexC; //hitPos/hitTexC/hitNormal：第一次命中信息
        
     
        // 在 view space 中，eye 位于原点，所以视线方向 = -posV
        float3 toEyeV = normalize(-posV);
        // 转换到 world space
        float3 toEyeW = normalize(mul(toEyeV, (float3x3) gInvView));
         // 法线从 view space 转到 world space
        float3 normalW = normalize(mul(normal, (float3x3) gInvView));
         // 计算反射方向（world space）
        float3 skyDirW = reflect(-toEyeW, normalW);
         // =========================================================
        
        hitLight = MultiReflectLight(
        posV, rayDir, skyDirW,gOcclusionRadius,
        isHit, hitPos, hitTexC, hitNormal); //用多反射函数

        // 3.3 预计算：未命中分支的基础数据（作为默认值）
        float3 q = posV + flip * gOcclusionRadius * offset; //计算射线终点（假设走满最大距离）,q 是 view-space 中的终点位置。
        float4 projQ = mul(float4(q, 1.0f), gProjTex); //将 q 从 view-space 投影到裁剪空间。
        projQ.xyz /= projQ.w; //透视除法，转为 NDC 坐标（[-1,1]）。
        projQ.xy = saturate(projQ.xy);
        
        //在 projQ.xy 处采样深度图，得到该屏幕位置的 view-space 深度 rz（负值）。
        float rz = NdcDepthToViewDepth(gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r);
        //作用：检查 q.z 是否远离相机平面（|q.z| ≥ gMinThreshold）。 若q 靠近 z = 0，则该 fallback无效
        float nonHitValid = step(gMinThreshold, abs(q.z)); // step(a, x)：若 x ≥ a 返回 1，否则 0。

        // 3.4 定义核心变量（默认=未命中值，命中时覆盖）
        float3 samplePos = q * (rz / q.z); //屏幕空间中该 UV 处的真实几何位置（view-space）。
        float distZ = posV.z - rz; //计算当前点与采样点的 view-space 深度差。
        
        //samplePos - posV：从当前点指向采样点的向量。
        //dot(normal, ...)：计算该方向与表面法线的对齐程度。
        //max(..., 0.0f)：只考虑半球内（正面）的贡献。
        float dp = max(dot(normal, normalize(samplePos - posV)), 0.0f);
        
        //保存采样点的 UV，用于后续采样反照率。
        float2 sampleTexC = projQ.xy;

        // 3.6 命中时覆盖核心变量
        if (isHit)
        {
            //若第一次命中，用真实命中数据覆盖 fallback 值
            //samplePos← hitPos
            //distZ← 真实深度差
            //dp← 真实方向权重
            //sampleTexC← 真实UV
            samplePos = hitPos;
            distZ = posV.z - hitPos.z;
            dp = max(dot(normal, normalize(hitPos - posV)), 0.0f);
            sampleTexC = hitTexC;
        }

        //// 3.7 有效性过滤，无效样本的贡献置零，确保累加时不干扰结果。
        //float validMask = (isHit ? 1.0f : nonHitValid);
        //distZ *= validMask;
        //dp *= validMask;
        //hitLight *= validMask;

        //// 3.8 累计结果
        //occlusionSum += dp * OcclusionFunction(distZ);
        //float indirectValid = step(-0.1f, distZ) * step(0.01f, dp);
        //// 过滤近距离采样 距离<0.05视为自身
        //float minSampleDist = step(0.05f, distance(samplePos, posV));
        //indirectValid *= minSampleDist;
        //indirectLightSum += hitLight * indirectValid * 2.0f;
        //validSampleCount += int(indirectValid * validMask);
        
        // 3.7 遮挡计算（仅用于 SSAO）
        float validMask = (isHit ? 1.0f : nonHitValid);
        occlusionSum += dp * OcclusionFunction(distZ) * validMask;

// 3.8 间接光有效性：命中时用几何验证，未命中时允许 sky 贡献
        float indirectValid;
        if (isHit)
        {
    // 命中：使用几何距离和方向验证
            float minSampleDist = step(0.05f, distance(samplePos, posV));
            indirectValid = step(-0.1f, distZ) * step(0.01f, dp) * minSampleDist;
        }
        else
        {
    // 未命中（sky）：只要射线有效，就允许贡献
            indirectValid = nonHitValid; // 即 q.z 不接近 0
        }

        indirectLightSum += hitLight * indirectValid * 2.0f;
        validSampleCount += int(indirectValid);
    }

    // 4. 归一化 + 最终结果计算
    float validFactor = validSampleCount > 0 ? 1.0f / validSampleCount : 0.0f;
    occlusionSum *= validFactor;
    indirectLightSum *= validFactor;

    float occlusion = saturate(pow(1.0f - occlusionSum, 2.0f));
    float3 ssgiColor = saturate(indirectLightSum + float3(0.15f, 0.15f, 0.15f));

    return float4(ssgiColor, occlusion);
}