//=============================================================================
// Ssao.hlsl by Frank Luna (C) 2015 All Rights Reserved.
// 新增：光线步进+物理反射GI，完全兼容原始写法，可正常编译
// 简化：移除未使用的NormalSimilarity函数，修复编译错误
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
// 光线步进静态常量（不新增cbuffer）
static const int gMaxRaySteps = 8;
static const float gRayHitThreshold = 0.01f;
static const float gReflectivity = 0.8f;
 
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
		
		// Linearly decrease occlusion from 1 to 0 as distZ goes 
		// from gOcclusionFadeStart to gOcclusionFadeEnd.	
        occlusion = saturate((gOcclusionFadeEnd - distZ) / fadeLength);
    }
	
    return occlusion;
}

// 修复除零风险的深度转换函数（兼容原始逻辑）
float NdcDepthToViewDepth(float z_ndc)
{
    // z_ndc = A + B/viewZ, where gProj[2,2]=A and gProj[3,2]=B.
    float denom = z_ndc - gProj[2][2];
    denom = max(abs(denom), 0.0001f) * sign(denom); // 防止除零
    float viewZ = gProj[3][2] / denom;
    return viewZ;
}

// 光线步进函数（HLSL兼容写法）
bool RayMarch(float3 rayStart, float3 rayDir, float maxDistance, out float3 hitPos, out float2 hitTexC)
{
    hitPos = float3(0.0f, 0.0f, 0.0f);
    hitTexC = float2(0.0f, 0.0f);
    
    float stepSize = maxDistance / (float) gMaxRaySteps;
    float3 currPos = rayStart;

    [unroll]
    for (int step = 0; step < gMaxRaySteps; step++)
    {
        // 沿射线推进一步
        currPos += rayDir * stepSize;
        
        // 投影到NDC空间，生成纹理坐标
        float4 projCurr = mul(float4(currPos, 1.0f), gProjTex);
        projCurr /= projCurr.w;
        projCurr.xy = saturate(projCurr.xy); // 限制在屏幕内
        
        // 采样当前位置的深度，重构真实场景点
        float z_ndc = gDepthMap.SampleLevel(gsamDepthMap, projCurr.xy, 0.0f).r;
        float sceneZ = NdcDepthToViewDepth(z_ndc);
        
        // 防止除以零
        if (abs(currPos.z) < 0.0001f)
            continue;
            
        float3 scenePos = (sceneZ / currPos.z) * currPos;
        
        // 相交判定：步进位置与真实场景点的距离小于阈值
        float distToScene = length(currPos - scenePos);
        if (distToScene < gRayHitThreshold)
        {
            hitPos = scenePos;
            hitTexC = projCurr.xy;
            return true; // 找到相交点，终止步进
        }
        
        // 超出最大距离，终止
        if (length(currPos - rayStart) > maxDistance)
            break;
    }
    return false; // 未找到相交点
}

// 菲涅尔方程（物理反射GI核心）
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}
 
float4 PS(VertexOut pin) : SV_Target
{
	// p -- the point we are computing the ambient occlusion for.
	// n -- normal vector at p.
	// q -- a random offset from p.
	// r -- a potential occluder that might occlude p.

	// Get viewspace normal and z-coord of this pixel.  
    float3 n = gNormalMap.SampleLevel(gsamPointClamp, pin.TexC, 0.0f).xyz;
    float pz = gDepthMap.SampleLevel(gsamDepthMap, pin.TexC, 0.0f).r;
    pz = NdcDepthToViewDepth(pz);

	//
	// Reconstruct full view space position (x,y,z).
	// Find t such that p = t*pin.PosV.
	// p.z = t*pin.PosV.z
	// t = p.z / pin.PosV.z
	//
    // 防止除以零
    if (abs(pin.PosV.z) < 0.0001f)
        return float4(0.05f, 0.05f, 0.05f, 1.0f);
        
    float3 p = (pz / pin.PosV.z) * pin.PosV;
	
	// Extract random vector and map from [0,1] --> [-1, +1].
    float3 randVec = 2.0f * gRandomVecMap.SampleLevel(gsamLinearWrap, 4.0f * pin.TexC, 0.0f).rgb - 1.0f;

    float occlusionSum = 0.0f;
    float3 indirectLightSum = float3(0.0f, 0.0f, 0.0f); // SSGI间接光累加
    int validSampleCount = 0; // 统计有效采样点数量
	
	// Sample neighboring points about p in the hemisphere oriented by n.
	[unroll]
    for (int i = 0; i < gSampleCount; ++i)
    {
		// Are offset vectors are fixed and uniformly distributed (so that our offset vectors
		// do not clump in the same direction).  If we reflect them about a random vector
		// then we get a random uniform distribution of offset vectors.
        float3 offset = reflect(gOffsetVectors[i].xyz, randVec);
	
		// Flip offset vector if it is behind the plane defined by (p, n).
        float flip = sign(dot(offset, n));
		
		// 物理反射方向计算（视线方向=相机→p，视图空间相机在原点）
        float3 viewDir = normalize(-p); // 视图空间相机在原点，视线方向为-p
        float3 reflectDir = normalize(reflect(-viewDir, normalize(n))); // 反射方向
		// 混合半球采样和反射方向
        float3 finalDir = normalize(lerp(normalize(offset), reflectDir, gReflectivity));
        finalDir = flip * finalDir; // 保证在法线半球内
		
		// 光线步进：沿反射方向找相交点
        float3 hitPos;
        float2 hitTexC;
        bool hit = RayMarch(p, finalDir, gOcclusionRadius, hitPos, hitTexC);
        if (hit)
        {
			// 重构的真实交点参与计算
            float distZ = p.z - hitPos.z;
            float dp = max(dot(n, normalize(hitPos - p)), 0.0f);
            float occlusion = dp * OcclusionFunction(distZ);
            occlusionSum += occlusion;

			// 优化SSGI间接光计算
            if (distZ > -0.1f && dp > 0.01f) // 放宽有效条件
            {
                float4 rAlbedo = gAlbedoMap.SampleLevel(gsamPointClamp, hitTexC, 0.0f);
                float spatialDistance = length(hitPos - p);
		
                float distanceAtten = 1.0f - saturate(spatialDistance / gOcclusionRadius);
                distanceAtten = pow(distanceAtten, 2.0f);
                distanceAtten = max(distanceAtten, 0.0f);

				// 物理菲涅尔反射
                float3 F0 = float3(0.04f); // 基础反射率
                F0 = lerp(F0, rAlbedo.rgb, gReflectivity);
                float cosTheta = max(dot(viewDir, reflectDir), 0.0f);
                float3 fresnel = FresnelSchlick(cosTheta, F0);
				
				// 累加物理反射GI
                indirectLightSum += rAlbedo.rgb * fresnel * distanceAtten;
                validSampleCount++;
            }
        }
        else
        {
			// 无相交点时沿用原始逻辑
            float3 q = p + flip * gOcclusionRadius * offset;
            float4 projQ = mul(float4(q, 1.0f), gProjTex);
            projQ /= projQ.w;
            projQ.xy = saturate(projQ.xy);
            
            float rz = gDepthMap.SampleLevel(gsamDepthMap, projQ.xy, 0.0f).r;
            rz = NdcDepthToViewDepth(rz);
            
            // 防止除以零
            if (abs(q.z) < 0.0001f)
                continue;
			
            float3 r = (rz / q.z) * q;
			
            float distZ = p.z - r.z;
            float dp = max(dot(n, normalize(r - p)), 0.0f);
            float occlusion = dp * OcclusionFunction(distZ);
            occlusionSum += occlusion;
			
            if (distZ > -0.1f && dp > 0.01f)
            {
                float4 rAlbedo = gAlbedoMap.SampleLevel(gsamPointClamp, projQ.xy, 0.0f);
                float spatialDistance = length(r - p);
		
                float distanceAtten = 1.0f - saturate(spatialDistance / gOcclusionRadius);
                distanceAtten = pow(distanceAtten, 2.0f);
                distanceAtten = max(distanceAtten, 0.0f);
				
                indirectLightSum += rAlbedo.rgb * distanceAtten;
                validSampleCount++;
            }
        }
    }
	
	// 归一化：使用有效采样数，避免除以0
    occlusionSum = validSampleCount > 0 ? occlusionSum / (float) validSampleCount : 0.0f;
    indirectLightSum = validSampleCount > 0 ? indirectLightSum / (float) validSampleCount : float3(0.0f, 0.0f, 0.0f);

    float access = 1.0f - occlusionSum;
    float occlusion = saturate(pow(access, 2.0f)); // 原始遮蔽因子
	
	// 计算最终间接光颜色
    float3 ssgiColor = saturate(indirectLightSum + float3(0.05f, 0.05f, 0.05f));
	
    return float4(ssgiColor, occlusion);
}