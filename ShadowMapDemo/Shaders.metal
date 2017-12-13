//
//  Shaders.metal
//  metalSkyBoxDemo
//
//  Created by Luke, Benjamin on 11/10/17.
//  Copyright © 2017 Luke, Benjamin. All rights reserved.
//

#include <metal_stdlib>
#import "ShaderTypes.h"
using namespace metal;

typedef struct
{
    float4 position [[attribute(kVertexAttributePosition)]];
    float3 normal    [[attribute(kVertexAttributeNormal)]];
} Vertex;


typedef struct
{
    // TODO: What does this position attribute tag do?
    // TODO: I am still unclear as to how the texCoords get interpolated across the triangle
    float4 position [[ position ]];
    float4 eyePosition;
    float4 lightPosition;
    float3 normal;
} VertexOut;

vertex VertexOut vertex_scene(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(kBufferIndexUniforms) ]],
                               constant MeshUniforms & meshUniforms [[ buffer(kMeshBufferIndexUniforms)]] ) {
    VertexOut out;
    
    // Calculate the position of our vertex in clip space and output for clipping and rasterization
    out.position = uniforms.projectionMatrix * uniforms.inverseViewMatrix * meshUniforms.modelMatrix * in.position;
    
    // Calculate the position of our vertex in eye space for lighting calculations
    out.eyePosition = uniforms.inverseViewMatrix * meshUniforms.modelMatrix * in.position;
    
    // Calculate the position of our vertex in light space for shadow calculations
    out.lightPosition = uniforms.inverseDirectionalLightMatrix * meshUniforms.modelMatrix * in.position;
    
    out.normal = meshUniforms.normalMatrix * in.normal;
    
    return out;
}

fragment float4 fragment_scene(VertexOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(kBufferIndexUniforms) ]],
                               constant MeshUniforms & meshUniforms [[ buffer(kMeshBufferIndexUniforms)]],
                               depth2d<float> shadowTexture [[texture(kShadowMapTexture)]],
                               sampler shadowSampler         [[sampler(kShadowMapSampler)]]) {

    float3 normal = normalize(in.normal);
    // The directional light direction from the uniform buffer is pointing from the light to the origin and we need the inverted vector
    float3 directionalLightDirection = -uniforms.directionalLightDirection;

    // Calculate the contribution of the directional light as a sum of diffuse and specular terms
    float nDotL = dot(normal, directionalLightDirection);
    float3 directionalContribution = float3(0);
    {
        float intensity = saturate(nDotL);
        float3 diffuseTerm = uniforms.directionalLightColor * intensity;
        float3 H = normalize(directionalLightDirection - float3(in.eyePosition));
        float nDotH = dot(normal, H);
        float specularIntensity = pow( saturate( nDotH ), meshUniforms.materialShininess );
        float3 specularTerm = uniforms.directionalLightColor * specularIntensity;
        directionalContribution = diffuseTerm + specularTerm;
    }

    // The ambient contribution, which is an approximation for global, indirect lighting, is
    // the product of the ambient light intensity multiplied by the material's reflectance
    float3 ambientContribution = uniforms.ambientLightColor;

    float3 color = float3(1);
    
    float2 textureCoordinates = (uniforms.biasMatrix * uniforms.orthoProjectionMatrix * in.lightPosition).xy;
    float shadowMapDepthValue = shadowTexture.sample(shadowSampler, textureCoordinates);
    float depthValue = (uniforms.orthoProjectionMatrix * in.lightPosition).z;
    
    float shadowMapBias = 0.005*tan(acos(clamp(nDotL, 0.0, 1.0))); // cosTheta is dot( n,l ), clamped between 0 and 1
    shadowMapBias = clamp(shadowMapBias, 0.0, 0.01);
    
    if (shadowMapDepthValue < depthValue-shadowMapBias) {
        float3 lightContributions = ambientContribution * 0.5;
        color = float3(color.xyz) * lightContributions;
    } else {
        float3 lightContributions = ambientContribution + directionalContribution;
        color = float3(color.xyz) * lightContributions;
    }
    
    // We use the color we just computed and the alpha channel of our
    // colorMap for this fragment's alpha value
    return float4(color.x, color.y, color.z, 1.0);
}

vertex float4 vertex_shadow(Vertex in [[stage_in]],
                              constant Uniforms & uniforms [[ buffer(kBufferIndexUniforms) ]],
                              constant MeshUniforms & meshUniforms [[ buffer(kMeshBufferIndexUniforms)]] ) {
    // Calculate the position of our vertex in clip space and output for clipping and rasterization
    float4 position = uniforms.orthoProjectionMatrix * uniforms.inverseDirectionalLightMatrix * meshUniforms.modelMatrix * in.position;
    return position;
}

