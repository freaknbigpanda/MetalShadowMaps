//
//  ShaderTypes.h
//  ShadowMapDemo
//
//  Created by Luke, Benjamin on 11/28/17.
//  Copyright © 2017 Luke, Benjamin. All rights reserved.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef enum BufferIndices
{
    kBufferIndexMeshVertecies = 0,
    kBufferIndexUniforms      = 1,
    kMeshBufferIndexUniforms  = 2
} BufferIndices;

typedef enum Textures
{
    kCubeMapTexture = 0,
    kShadowMapTexture = 1
} Textures;

typedef enum Samplers
{
    kCubeMapSampler = 0,
    kShadowMapSampler = 1
} Samplers;

typedef enum VertexAttributes
{
    kVertexAttributePosition  = 0,
    kVertexAttributeNormal    = 1,
} VertexAttributes;

typedef struct
{
    // Per mesh uniforms
    float materialShininess;
    matrix_float4x4 modelMatrix;
    matrix_float3x3 normalMatrix;
} MeshUniforms;

typedef struct
{
    // Per Frame Uniforms
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 orthoProjectionMatrix;
    matrix_float4x4 viewMatrix;
    matrix_float4x4 inverseViewMatrix;
    
    // used to convert from homogeneous coordinates to 0-1 uv texture coordinates
    matrix_float4x4 biasMatrix;
    
    // Per Light Properties
    vector_float3 ambientLightColor;
    // Directional light direction points from the light to the origin 0,0 
    vector_float3 directionalLightDirection;
    vector_float3 directionalLightColor;
    
    matrix_float4x4 inverseDirectionalLightMatrix;
} Uniforms;

#endif /* ShaderTypes_h */
