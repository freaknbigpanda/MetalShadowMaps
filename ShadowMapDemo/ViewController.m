//
//  ViewController.m
//  ShadowMapDemo
//
//  Created by Luke, Benjamin on 11/28/17.
//  Copyright © 2017 Luke, Benjamin. All rights reserved.
//

#import "ViewController.h"
#import <MetalKit/MetalKit.h>
#import "Renderer.h"

@interface ViewController ()

@end

@implementation ViewController
{
    MTKView *_view;
    id<MTLDevice> _device;
    Renderer *_renderer;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Set the view to use the default device
    _device = MTLCreateSystemDefaultDevice();
    _view = (MTKView *)self.view;
    _view.delegate = self;
    _view.device = _device;
    
    _renderer = [[Renderer alloc] initWithMetalDevice:_device
                            renderDestinationProvider:self];
    
    [_renderer drawRectResized:_view.bounds.size];
    
    if(!_device)
    {
        NSLog(@"Metal is not supported on this device");
        self.view = [[UIView alloc] initWithFrame:self.view.frame];
    }
}

// Called whenever view changes orientation or layout is changed
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    [_renderer drawRectResized:view.bounds.size];
}

// Called whenever the view needs to render
- (void)drawInMTKView:(nonnull MTKView *)view
{
    @autoreleasepool
    {
        [_renderer update];
    }
}

// Methods to get and set state of the our ultimate render destination (i.e. the drawable)
# pragma mark RenderDestinationProvider implementation

- (MTLRenderPassDescriptor*) currentRenderPassDescriptor
{
    return _view.currentRenderPassDescriptor;
}

- (MTLPixelFormat) colorPixelFormat
{
    return _view.colorPixelFormat;
}

- (void) setColorPixelFormat: (MTLPixelFormat) pixelFormat
{
    _view.colorPixelFormat = pixelFormat;
}

- (MTLPixelFormat) depthStencilPixelFormat
{
    return _view.depthStencilPixelFormat;
}

- (void) setDepthStencilPixelFormat: (MTLPixelFormat) pixelFormat
{
    _view.depthStencilPixelFormat = pixelFormat;
}

- (NSUInteger) sampleCount
{
    return _view.sampleCount;
}

- (void) setSampleCount:(NSUInteger) sampleCount
{
    _view.sampleCount = sampleCount;
}

- (id<MTLDrawable>) currentDrawable
{
    return _view.currentDrawable;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
