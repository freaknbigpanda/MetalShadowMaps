//
//  ViewController.h
//  ShadowMapDemo
//
//  Created by Luke, Benjamin on 11/28/17.
//  Copyright © 2017 Luke, Benjamin. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "Renderer.h"

@interface ViewController : UIViewController <MTKViewDelegate, RenderDestinationProvider>


@end

