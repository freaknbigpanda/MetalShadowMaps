//
//  Renderer.m
//  ShadowMapDemo
//
//  Created by Luke, Benjamin on 11/28/17.
//  Copyright © 2017 Luke, Benjamin. All rights reserved.
//

#import "Renderer.h"
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>
#import "ShaderTypes.h"
#import "AAPLTransforms.h"

using namespace AAPL;

@implementation Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    id <MTLLibrary> _defaultLibrary;
    
    // Metal objects
    id <MTLBuffer> _dynamicUniformBuffer;
    id <MTLBuffer> _dynamicMeshUniformBuffer;
    id <MTLRenderPipelineState> _shadowPipeline;
    id <MTLRenderPipelineState> _shadowScenePipeline;
    id <MTLTexture> _shadowMap;
    id <MTLTexture> _crazyTexture;
    id <MTLSamplerState> _shadowMapSamplerState;
    // TODO: Depth and stencil buffers
    id <MTLDepthStencilState> _shadowMapDepthState;
    id <MTLDepthStencilState> _depthState;
    
    // Metal vertex descriptor specifying how vertices will by laid out for input into our render
    //   pipeline and how we'll layout our Model IO verticies
    MTLVertexDescriptor *_mtlVertexDescriptor;
    
    // The object controlling the ultimate render destination
    __weak id<RenderDestinationProvider> _renderDestination;
    
    // Offset within _dynamicUniformBuffer to set for the current frame
    uint32_t _uniformBufferOffset;
    
    // Used to determine _uniformBufferStride each frame.
    //   This is the current frame number modulo kMaxBuffersInFlight
    uint8_t _uniformBufferIndex;
    
    // Address to write dynamic uniforms to each frame
    void* _uniformBufferAddress;
    
    // Offset within _dynamicUniformBuffer to set for the current frame
    uint32_t _meshUniformBufferOffset;
    
    // Used to determine _uniformBufferStride each frame.
    //   This is the current frame number modulo kMaxBuffersInFlight
    uint8_t _meshUniformBufferIndex;
    
    // Address to write dynamic uniforms to each frame
    void* _meshUniformBufferAddress;
    
    // Projection matrix calculated as a function of view size
    matrix_float4x4 _projectionMatrix;
    
    // lighting control
    bool _shouldRotateLight;
    vector_float3 _xAxisLightRotation;
    vector_float3 _yAxisLightRotation;
    uint16_t _timeForFullRotation; // time in seconds
    vector_float3 _lightDirection; // ignored for rotating light source
    CFTimeInterval _firstLightUpdate;
    
    // Meshes
    
    // environment
    MTKMesh *_sphere;
    MTKMesh *_cylinder;
    MTKMesh *_box;
    MTKMesh *_ground;
}

static const NSUInteger kNumberOfMeshes = 4;

static const NSUInteger kSphereIndex = 0;
static const NSUInteger kCylinderIndex = 1;
static const NSUInteger kBoxIndex = 2;
static const NSUInteger kGroundIndex = 3;

// The max number of command buffers in flight
static const NSUInteger kMaxBuffersInFlight = 3;

// The 256 byte aligned size of our uniform structure
static const size_t kAlignedUniformsSize = (sizeof(Uniforms) & ~0xFF) + 0x100;

// The 256 byte aligned size of our uniform structure
static const size_t kAlignedMeshUniformsSize = ((sizeof(MeshUniforms)*kNumberOfMeshes) & ~0xFF) + 0x100;

-(nonnull instancetype) initWithMetalDevice:(nonnull id<MTLDevice>)device
                  renderDestinationProvider:(nonnull id<RenderDestinationProvider>)renderDestinationProvider
{
    self = [super init];
    if(self)
    {
        _device = device;
        _renderDestination = renderDestinationProvider;
        _inFlightSemaphore = dispatch_semaphore_create(kMaxBuffersInFlight);
        [self _loadMetal];
        [self _loadAssets];
    }
    
    return self;
}

- (void)_loadMetal
{
    // Create and load our basic Metal state objects
    
    // Load all the shader files with a metal file extension in the project
    _defaultLibrary = [_device newDefaultLibrary];
    
    // Calculate our uniform buffer size.  We allocate kMaxBuffersInFlight instances for uniform
    //   storage in a single buffer.  This allows us to update uniforms in a ring (i.e. triple
    //   buffer the uniforms) so that the GPU reads from one slot in the ring wil the CPU writes
    //   to another.  Also uniform storage must be aligned (to 256 bytes) to meet the requirements
    //   to be an argument in the constant address space of our shading functions.
    NSUInteger uniformBufferSize = kAlignedUniformsSize * kMaxBuffersInFlight;
    NSUInteger meshUniformBufferSize = kAlignedMeshUniformsSize * kMaxBuffersInFlight;
    
    
    _dynamicMeshUniformBuffer = [_device newBufferWithLength:meshUniformBufferSize
                                                 options:MTLResourceStorageModeShared];
    _dynamicUniformBuffer = [_device newBufferWithLength:uniformBufferSize
                                                 options:MTLResourceStorageModeShared];
    
    _dynamicMeshUniformBuffer.label = @"UniformMeshBuffer";
    _dynamicUniformBuffer.label = @"UniformBuffer";
    
    // Create a vertex descriptor for our Metal pipeline. Specifies the layout of vertices the
    //   pipeline should expect.  The layout below keeps attributes used to calculate vertex shader
    //   output position separate (world position, skinning, tweening weights) separate from other
    //   attributes (texture coordinates, normals).  This generally maximizes pipeline efficiency
    
    _mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];
    
    // Positions.
    _mtlVertexDescriptor.attributes[kVertexAttributePosition].format = MTLVertexFormatFloat4;
    _mtlVertexDescriptor.attributes[kVertexAttributePosition].offset = 0;
    _mtlVertexDescriptor.attributes[kVertexAttributePosition].bufferIndex = kBufferIndexMeshVertecies;
    
    // Normals.
    //TODO: Try to use a smaller format here, I don't think we need full float percision for normals
    _mtlVertexDescriptor.attributes[kVertexAttributeNormal].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[kVertexAttributeNormal].offset = sizeof(simd_float4);
    _mtlVertexDescriptor.attributes[kVertexAttributeNormal].bufferIndex = kBufferIndexMeshVertecies;
    
    // Position Buffer Layout
    _mtlVertexDescriptor.layouts[kBufferIndexMeshVertecies].stride = sizeof(simd_float4) + sizeof(simd_float4);
    _mtlVertexDescriptor.layouts[kBufferIndexMeshVertecies].stepRate = 1;
    _mtlVertexDescriptor.layouts[kBufferIndexMeshVertecies].stepFunction = MTLVertexStepFunctionPerVertex;
    
    // TODO: Depth and stencil buffers
    _renderDestination.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    _renderDestination.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    _renderDestination.sampleCount = 1;
    
    NSError *error = NULL;
    
    // Shadow map scene pipeline setup
    MTLRenderPipelineDescriptor * shadowScenePipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    
    id <MTLFunction> shadowSceneVertexFunction = [_defaultLibrary newFunctionWithName:@"vertex_scene"];
    id <MTLFunction> shadowSceneFragmentFunction = [_defaultLibrary newFunctionWithName:@"fragment_scene"];
    
    shadowScenePipelineStateDescriptor.label = @"ShadowScenePipeline";
    shadowScenePipelineStateDescriptor.sampleCount = _renderDestination.sampleCount;
    shadowScenePipelineStateDescriptor.vertexFunction = shadowSceneVertexFunction;
    shadowScenePipelineStateDescriptor.fragmentFunction = shadowSceneFragmentFunction;
    shadowScenePipelineStateDescriptor.vertexDescriptor = _mtlVertexDescriptor;
    shadowScenePipelineStateDescriptor.colorAttachments[0].pixelFormat = _renderDestination.colorPixelFormat;
    shadowScenePipelineStateDescriptor.depthAttachmentPixelFormat = _renderDestination.depthStencilPixelFormat;
    
    _shadowScenePipeline = [_device newRenderPipelineStateWithDescriptor:shadowScenePipelineStateDescriptor error:&error];
    
    if (!_shadowScenePipeline)
    {
        NSLog(@"Failed to created pipeline state, error %@", error);
    }
    
    // Shadow map scene pipeline setup
    MTLRenderPipelineDescriptor * shadowPipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    
    id <MTLFunction> shadowVertexFunction = [_defaultLibrary newFunctionWithName:@"vertex_shadow"];
    
    shadowPipelineStateDescriptor.label = @"ShadowPipeline";
    shadowPipelineStateDescriptor.sampleCount = 1;
    shadowPipelineStateDescriptor.vertexFunction = shadowVertexFunction;
    shadowPipelineStateDescriptor.vertexDescriptor = _mtlVertexDescriptor;
    shadowPipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    
    _shadowPipeline = [_device newRenderPipelineStateWithDescriptor:shadowPipelineStateDescriptor error:&error];
    
    if (!_shadowPipeline)
    {
        NSLog(@"Failed to created pipeline state, error %@", error);
    }
    
    // Texture samplers
    MTLSamplerDescriptor *samplerDescriptor = [MTLSamplerDescriptor new];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterNearest;
    samplerDescriptor.compareFunction = MTLCompareFunctionLess;
    _shadowMapSamplerState = [_device newSamplerStateWithDescriptor:samplerDescriptor];
    
    // Create the command queue
    _commandQueue = [_device newCommandQueue];
    
    // Depth stencil states
    //TODO: I *still* do not really understand what these do
    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStateDesc.depthWriteEnabled = YES;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];
    
    MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
    desc.depthCompareFunction = MTLCompareFunctionLessEqual;
    desc.depthWriteEnabled = YES;
    
    MTLStencilDescriptor *stencilStateDesc = [[MTLStencilDescriptor alloc] init];
    stencilStateDesc.stencilCompareFunction = MTLCompareFunctionAlways;
    stencilStateDesc.stencilFailureOperation = MTLStencilOperationKeep;

    desc.frontFaceStencil = stencilStateDesc;
    desc.backFaceStencil = stencilStateDesc;
    _shadowMapDepthState = [_device newDepthStencilStateWithDescriptor: depthStateDesc];
}

- (void)_loadAssets
{
    // Create a MetalKit mesh buffer allocator so that ModelIO will load mesh data directly into
    //   Metal buffers accessible by the GPU
    MTKMeshBufferAllocator *metalAllocator = [[MTKMeshBufferAllocator alloc]
                                              initWithDevice: _device];
    
    // TODO: Figure out why the plane doesn't work
    MDLMesh * ground = [MDLMesh newBoxWithDimensions:(vector_float3) {50, 50, 1} segments:(vector_uint3) {100, 100, 1} geometryType:MDLGeometryTypeTriangles inwardNormals:NO allocator:metalAllocator];
    
    MDLMesh * box = [MDLMesh newBoxWithDimensions:(vector_float3) {10, 10, 10} segments:(vector_uint3) {10, 10, 10} geometryType:MDLGeometryTypeTriangles inwardNormals:NO allocator:metalAllocator];
    
    MDLMesh * cylinder = [MDLMesh newCylinderWithHeight:10 radii:(vector_float2) {5, 5} radialSegments:30 verticalSegments:20 geometryType:MDLGeometryTypeTriangles inwardNormals:NO allocator:metalAllocator];
    
    MDLMesh * sphere = [MDLMesh newEllipsoidWithRadii:5 radialSegments:360 verticalSegments:360 geometryType:MDLGeometryTypeTriangles inwardNormals:NO hemisphere:NO allocator:metalAllocator];
    
    // Creata a Model IO vertexDescriptor so that we format/layout our model IO mesh vertices to
    //   fit our Metal render pipeline's vertex descriptor layout
    MDLVertexDescriptor *mdlVertexDescriptor =
    MTKModelIOVertexDescriptorFromMetal(_mtlVertexDescriptor);
    
    // Indicate how each Metal vertex descriptor attribute maps to each ModelIO attribute
    mdlVertexDescriptor.attributes[kVertexAttributePosition].name  = MDLVertexAttributePosition;
    mdlVertexDescriptor.attributes[kVertexAttributeNormal].name    = MDLVertexAttributeNormal;
    
    sphere.vertexDescriptor = mdlVertexDescriptor;
    box.vertexDescriptor = mdlVertexDescriptor;
    ground.vertexDescriptor = mdlVertexDescriptor;
    cylinder.vertexDescriptor = mdlVertexDescriptor;
    
    // Create MetalKit meshs (and submeshes) backed by Metal buffers
    _sphere = [self mtkMeshWithMdlMesh:sphere];
    _ground = [self mtkMeshWithMdlMesh:ground];
    _box = [self mtkMeshWithMdlMesh:box];
    _cylinder = [self mtkMeshWithMdlMesh:cylinder];
    
    // inital light settings
    _xAxisLightRotation = (vector_float3) { 1.0, 0.0, 0.0 };
    _yAxisLightRotation = (vector_float3) { 0.0, 1.0, 0.0 };
    _timeForFullRotation = 1;
    _shouldRotateLight = YES;
    _firstLightUpdate = CACurrentMediaTime();
}

- (void)buildDepthTextureForDrawableSize:(CGSize)drawableSize
{
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                          width:2048
                                                                                         height:2048
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _shadowMap = [_device newTextureWithDescriptor:descriptor];
    [_shadowMap setLabel:@"Custom Depth Texture"];
}

- (MTKMesh *) mtkMeshWithMdlMesh:(MDLMesh *)mesh {
    
    NSError * error;
    MTKMesh * returnMesh = [[MTKMesh alloc] initWithMesh:mesh
                                     device:_device
                                      error:&error];
    
    if(!returnMesh || error)
    {
        NSLog(@"Error creating MetalKit mesh %@", error.localizedDescription);
        return nil;
    }
    
    return returnMesh;
}

- (void)drawRectResized:(CGSize)size
{
    // When reshape is called, update the aspect ratio and projection matrix since the view
    //   orientation or size has changed
    float aspect = size.width / (float)size.height;
    _projectionMatrix = perspective_fov(65.0f, aspect, 0.1f, 100.0f);
    
    CGFloat scale = [UIScreen mainScreen].scale;
    [self buildDepthTextureForDrawableSize:CGSizeMake(size.width*scale, size.height*scale)];
}

- (vector_float3) getNextDirectionalLightDirection {
    
    if (_shouldRotateLight) {
        CFTimeInterval currentTime = CACurrentMediaTime();
        CFTimeInterval timePassed = currentTime - _firstLightUpdate;
        
        CGFloat fractionOfFullRotation = timePassed/_timeForFullRotation;
        CGFloat rotationInRads = fractionOfFullRotation * M_2_PI;
        
        vector_float3 xDirection = _xAxisLightRotation * cos(rotationInRads);
        vector_float3 yDirection = _yAxisLightRotation * sin(rotationInRads);
        
        return xDirection + yDirection;
    } else {
        return _lightDirection;
    }
}

- (void)_updateDynamicBufferState
{
    // Update the location(s) to which we'll write to in our dynamically changing Metal buffers for
    //   the current frame (i.e. update our slot in the ring buffer used for the current frame)
    
    _uniformBufferIndex = (_uniformBufferIndex + 1) % kMaxBuffersInFlight;
    
    _uniformBufferOffset = kAlignedUniformsSize * _uniformBufferIndex;
    
    _uniformBufferAddress = ((uint8_t*)_dynamicUniformBuffer.contents) + _uniformBufferOffset;
    
    _meshUniformBufferIndex = (_meshUniformBufferIndex + 1) % kMaxBuffersInFlight;
    
    _meshUniformBufferOffset = kAlignedMeshUniformsSize * _meshUniformBufferIndex;
    
    _meshUniformBufferAddress = ((uint8_t*)_dynamicMeshUniformBuffer.contents) + _meshUniformBufferOffset;
}

- (void)_updateGameState
{
    /// Update any game state before encoding renderint commands to our drawable
    
    Uniforms * uniforms = (Uniforms*)_uniformBufferAddress;
    
    // lighting
    vector_float3 ambientLightColor = {0.02, 0.02, 0.02};
    uniforms->ambientLightColor = ambientLightColor;
    
    vector_float3 directionalLightDirection = [self getNextDirectionalLightDirection];
    directionalLightDirection = simd_normalize(directionalLightDirection);
    uniforms->directionalLightDirection = directionalLightDirection;
    
    // the directional light direction is the -z axis = -uniforms->directionalLightDirection
    // the y axis is always going to be +z axis in world space = (vector_float3) {0,0,1}
    // the x axis is going to be the tricky one but it should just be simd_cross(up, zAxis);
    
    vector_float3 zAxis = (vector_float3) {directionalLightDirection.x * -1, directionalLightDirection.y * -1, directionalLightDirection.z * -1};
    zAxis = simd_normalize(zAxis);
    vector_float3 yAxis = (vector_float3) {0.0,0.0,1.0};
    vector_float3 xAxis = simd_normalize(simd_cross(yAxis, zAxis));
    
    matrix_float4x4 lightOrientationMatrix = matrix_make((vector_float4) {xAxis.x, xAxis.y, xAxis.z, 0},
                                                         (vector_float4) {yAxis.x, yAxis.y, yAxis.z, 0},
                                                         (vector_float4) {zAxis.x, zAxis.y, zAxis.z, 0},
                                                         (vector_float4) { 0,0,0,1 });

    uniforms->inverseDirectionalLightMatrix = simd_inverse(lightOrientationMatrix);
    
    vector_float3 directionalLightColor = {.7, .7, .7};
    uniforms->directionalLightColor = directionalLightColor;;
    
    uniforms->projectionMatrix = _projectionMatrix;
    
    // left right bottom top near far
    uniforms->orthoProjectionMatrix = ortho2d_oc(-30.0f, 30.0f, -30.0f, 30.0f, 30.0f, -30.0f);
    
    uniforms->viewMatrix = translate(0.0, 20.0, 60.0);
    
    // Notes: Inverting a matrix causes the order of the operations applied to the point or vector to also be swapped
    // Notes: Applying a rotation and then a translation results in a translation along the original axis specified but rotated by the rotation
    
    uniforms->inverseViewMatrix = simd_inverse(uniforms->viewMatrix);
    
    // 1/2 scale and 0.5 translation to move homogeneous -1,1 coordinates to 0,1 texture coordinates
    // TODO: Why do we have to have a -0.5 scale on y for the bias matrix? Could it be that texture coordinates in metal are weird?
    uniforms->biasMatrix = matrix_multiply(translate(0.5, 0.5, 0.0), scale(0.5, -0.5, 0.0));
    
    MeshUniforms * meshUniforms = (MeshUniforms*)_meshUniformBufferAddress;
    
    MeshUniforms * groundUniform = &meshUniforms[kGroundIndex];
    groundUniform->materialShininess = 30;
    groundUniform->modelMatrix = rotate(-90, (vector_float3) {1.0,0.0,0.0});
    groundUniform->modelMatrix = matrix_multiply(rotate(30, (vector_float3) {0.0,0.0,1.0} ), groundUniform->modelMatrix);
    matrix_float4x4 modelViewMatrix = matrix_multiply(uniforms->inverseViewMatrix, groundUniform->modelMatrix);
    groundUniform->normalMatrix = matrix3x3_upper_left(modelViewMatrix);
    groundUniform->normalMatrix = matrix_invert(matrix_transpose(groundUniform->normalMatrix));
    
    MeshUniforms * sphereUniform = &meshUniforms[kSphereIndex];
    sphereUniform->materialShininess = 30;
    sphereUniform->modelMatrix = translate(10.0, 15.0, 0.0);
    modelViewMatrix = matrix_multiply(uniforms->inverseViewMatrix, sphereUniform->modelMatrix);
    sphereUniform->normalMatrix = matrix3x3_upper_left(modelViewMatrix);
    sphereUniform->normalMatrix = matrix_invert(matrix_transpose(sphereUniform->normalMatrix));
    
    MeshUniforms * boxUniform = &meshUniforms[kBoxIndex];
    boxUniform->materialShininess = 30;
    boxUniform->modelMatrix = translate(-20.0, 15.0, 0.0);
    modelViewMatrix = matrix_multiply(uniforms->inverseViewMatrix, boxUniform->modelMatrix);
    boxUniform->normalMatrix = matrix3x3_upper_left(modelViewMatrix);
    boxUniform->normalMatrix = matrix_invert(matrix_transpose(boxUniform->normalMatrix));
    
    MeshUniforms * cylinderUniform = &meshUniforms[kCylinderIndex];
    cylinderUniform->materialShininess = 30;
    cylinderUniform->modelMatrix = translate(0.0, 15.0, 17.0);
    modelViewMatrix = matrix_multiply(uniforms->inverseViewMatrix, cylinderUniform->modelMatrix);
    cylinderUniform->normalMatrix = matrix3x3_upper_left(modelViewMatrix);
    cylinderUniform->normalMatrix = matrix_invert(matrix_transpose(cylinderUniform->normalMatrix));
}

- (MTLRenderPassDescriptor *)createRenderPassDescriptorForShadowMap
{
    
    MTLRenderPassDescriptor *renderPass = [MTLRenderPassDescriptor new];
    renderPass.depthAttachment.texture = _shadowMap;
    renderPass.depthAttachment.loadAction = MTLLoadActionClear;
    renderPass.depthAttachment.storeAction = MTLStoreActionStore;
    renderPass.depthAttachment.clearDepth = 1.0;
    
    return renderPass;
}

- (void) update {
    
    // Wait to ensure only kMaxBuffersInFlight are getting proccessed by any stage in the Metal
    //   pipeline (App, Metal, Drivers, GPU, etc)
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);
    
    // Create a new command buffer for each renderpass to the current drawable
    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer]; // created once per frame
    commandBuffer.label = @"MyCommand";
    
    // Add completion hander which signal _inFlightSemaphore when Metal and the GPU has fully
    //   finished proccssing the commands we're encoding this frame.  This indicates when the
    //   dynamic buffers, that we're writing to this frame, will no longer be needed by Metal
    //   and the GPU.
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(block_sema);
     }];
    
    [self _updateDynamicBufferState];
    [self _updateGameState];
    
    [self _renderShadowMapWithCommandBuffer:commandBuffer];
    [self _renderSceneWithCommandBuffer:commandBuffer];
    
    // Schedule a present once the framebuffer is complete using the current drawable
    [commandBuffer presentDrawable:_renderDestination.currentDrawable];
    
    // Finalize rendering here & push the command buffer to the GPU
    [commandBuffer commit];
}

- (void) _renderShadowMapWithCommandBuffer:(id <MTLCommandBuffer>)commandBuffer {
    
    // Obtain a renderPassDescriptor generated from the view's drawable textures
    MTLRenderPassDescriptor* renderPassDescriptor = [self createRenderPassDescriptorForShadowMap];
    
    if(renderPassDescriptor != nil) {
        
        // Create a render command encoder so we can render into something
        id <MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"ShadowMapEncoder";
        [renderEncoder setCullMode:MTLCullModeFront];
        
        [renderEncoder setFragmentTexture:_shadowMap atIndex:kShadowMapTexture];
        
        [self _renderMeshesWithRenderEncoder:renderEncoder depthState:_shadowMapDepthState andPipelineState:_shadowPipeline];
        
        // We're done encoding commands
        [renderEncoder endEncoding];
    }
}

- (void) _renderSceneWithCommandBuffer:(id <MTLCommandBuffer>)commandBuffer {
    // Obtain a renderPassDescriptor generated from the view's drawable textures
    MTLRenderPassDescriptor* renderPassDescriptor = _renderDestination.currentRenderPassDescriptor;
    
    if(renderPassDescriptor != nil) {
        
        // Create a render command encoder so we can render into something
        id <MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"ShadowSceneEncoder";
        
        [renderEncoder setFragmentTexture:_shadowMap atIndex:kShadowMapTexture];
        [renderEncoder setFragmentSamplerState:_shadowMapSamplerState atIndex:kShadowMapSampler];
        [renderEncoder setCullMode:MTLCullModeBack];
        
        [self _renderMeshesWithRenderEncoder:renderEncoder depthState:_depthState andPipelineState:_shadowScenePipeline];
        
        // We're done encoding commands
        [renderEncoder endEncoding];
    }
}

- (void) _renderMeshesWithRenderEncoder:(id <MTLRenderCommandEncoder>)renderEncoder depthState:(id <MTLDepthStencilState>)depthState andPipelineState:(id <MTLRenderPipelineState>)pipelineState {
    // Push a debug group allowing us to identify render commands in the GPU Frame Capture tool
    [renderEncoder pushDebugGroup:@"DrawShadowScene"];
    
    [renderEncoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [renderEncoder setRenderPipelineState:pipelineState];
    [renderEncoder setDepthStencilState:depthState];
    
    // Set any buffers fed into our render pipeline
    [renderEncoder setVertexBuffer:_dynamicUniformBuffer
                            offset:_uniformBufferOffset
                           atIndex:kBufferIndexUniforms];
    
    [renderEncoder setFragmentBuffer:_dynamicUniformBuffer
                              offset:_uniformBufferOffset
                             atIndex:kBufferIndexUniforms];

    [renderEncoder setFragmentBuffer:_dynamicMeshUniformBuffer
                              offset:_meshUniformBufferOffset
                             atIndex:kMeshBufferIndexUniforms];

    [renderEncoder setVertexBuffer:_dynamicMeshUniformBuffer
                            offset:_meshUniformBufferOffset
                           atIndex:kMeshBufferIndexUniforms];
    
    [renderEncoder setVertexBufferOffset:_meshUniformBufferOffset + (kSphereIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];
    [renderEncoder setFragmentBufferOffset:_meshUniformBufferOffset + (kSphereIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];

    [self renderMesh:_sphere withRenderEncoder:renderEncoder];
    
    [renderEncoder setVertexBufferOffset:_meshUniformBufferOffset + (kGroundIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];
    [renderEncoder setFragmentBufferOffset:_meshUniformBufferOffset + (kGroundIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];

    [self renderMesh:_ground withRenderEncoder:renderEncoder];
    
    [renderEncoder setVertexBufferOffset:_meshUniformBufferOffset + (kBoxIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];
    [renderEncoder setFragmentBufferOffset:_meshUniformBufferOffset + (kBoxIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];

    [self renderMesh:_box withRenderEncoder:renderEncoder];

    [renderEncoder setVertexBufferOffset:_meshUniformBufferOffset + (kCylinderIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];
    [renderEncoder setFragmentBufferOffset:_meshUniformBufferOffset + (kCylinderIndex * sizeof(MeshUniforms))
                                 atIndex:kMeshBufferIndexUniforms];

    [self renderMesh:_cylinder withRenderEncoder:renderEncoder];
    
    [renderEncoder popDebugGroup];
}

- (void) renderMesh:(MTKMesh *)mesh withRenderEncoder:(id <MTLRenderCommandEncoder>)renderEncoder {
    
    for (NSUInteger bufferIndex = 0; bufferIndex < mesh.vertexBuffers.count; bufferIndex++)
    {
        MTKMeshBuffer *vertexBuffer = mesh.vertexBuffers[bufferIndex];
        if((NSNull*)vertexBuffer != [NSNull null])
        {
            [renderEncoder setVertexBuffer:vertexBuffer.buffer
                                    offset:vertexBuffer.offset
                                   atIndex:bufferIndex];
        }
    }
    
    // Draw each submesh of our mesh
    for(MTKSubmesh *submesh in mesh.submeshes)
    {
        [renderEncoder drawIndexedPrimitives:submesh.primitiveType
                                  indexCount:submesh.indexCount
                                   indexType:submesh.indexType
                                 indexBuffer:submesh.indexBuffer.buffer
                           indexBufferOffset:submesh.indexBuffer.offset];
    }
    
}

//TODO: Convert everything to use C++ classes - I don't want any C at all in here.

matrix_float4x4 __attribute__((__overloadable__))
matrix_make(float m00, float m10, float m20, float m30,
            float m01, float m11, float m21, float m31,
            float m02, float m12, float m22, float m32,
            float m03, float m13, float m23, float m33)
{
    return (matrix_float4x4){ {
        { m00, m10, m20, m30 },
        { m01, m11, m21, m31 },
        { m02, m12, m22, m32 },
        { m03, m13, m23, m33 } } };
}

matrix_float3x3 __attribute__((__overloadable__))
matrix_make(vector_float3 col0, vector_float3 col1, vector_float3 col2)
{
    return (matrix_float3x3){ col0, col1, col2 };
}

matrix_float4x4 __attribute__((__overloadable__))
matrix_make(vector_float4 col0, vector_float4 col1, vector_float4 col2, vector_float4 col3)
{
    return (matrix_float4x4){ col0, col1, col2, col3 };
}

matrix_float3x3 matrix3x3_upper_left(matrix_float4x4 m)
{
    vector_float3 x = m.columns[0].xyz;
    vector_float3 y = m.columns[1].xyz;
    vector_float3 z = m.columns[2].xyz;
    return matrix_make(x, y, z);
}
@end

