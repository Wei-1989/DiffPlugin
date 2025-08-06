import numpy

from modules import SCBNet
from modules.unet.My_Unet2DConditionModel import FeatureFusion
from modules.myVAE import RefineModule
from modules.vae_FeatureFusionBlock import MultiScaleFeatureFusionModule
import torch
import torch.nn as nn
def analyse(model, input):
    import time
    # from fvcore.nn import FlopCountAnalysis
    from thop import profile
    import numpy
    time_list = []
    for i in range(11):
        start_time = time.time()
        model(*input)
        end_time = time.time()
        time_list.append(end_time - start_time)
    cons_time = numpy.mean(time_list[1:])
    print(f"Inference time: {cons_time * 1000:.2f} ms")

    # flops = FlopCountAnalysis(model, input)
    # print(f"Total FLOPs: {flops.total() / 1e9:.6f} GFLOPs")
    macs, params = profile(model, input)
    print('MACs:', macs / 1e9, 'G')
    print(f"Total parameters: {params / 1e6:,} M")

    return cons_time * 1000,macs / 1e9,params / 1e6
HEIGHT=256
h=HEIGHT/8


class fusion_module(nn.Module):
    def __init__(self):
        super(fusion_module, self).__init__()
        self.fusion = nn.ModuleList([
            FeatureFusion(in_channels=320),
            FeatureFusion(in_channels=320),
            FeatureFusion(in_channels=320),
            FeatureFusion(in_channels=320),
            FeatureFusion(in_channels=640),
            FeatureFusion(in_channels=640),
            FeatureFusion(in_channels=640),
            FeatureFusion(in_channels=1280),
            FeatureFusion(in_channels=1280),
            FeatureFusion(in_channels=1280),
            FeatureFusion(in_channels=1280),
            FeatureFusion(in_channels=1280),
        ])

    def forward(self, x, y):
        for i,j,k in zip(x,y,self.fusion):
            k(i,j)

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.refine_module_list=nn.ModuleList(
                [
                    RefineModule(512,512,1),
                    RefineModule(512,512,2),
                    RefineModule(256,512,2),
                    RefineModule(128,256,2),
                ]
            )
        self.final_refine=RefineModule(128, 128, 1)
    def forward(self,x_list,y_list):
        for refine_module,x,y in zip(self.refine_module_list,x_list,y_list):
            sample = refine_module(y, x)



timestep=torch.tensor(1).cuda()
cond_img=torch.rand(1,8,int(HEIGHT/8),int(HEIGHT/8)).cuda()
x_unet_list=[
    torch.rand(1, 320, int(h), int(h)).cuda(),
    torch.rand(1, 320, int(h), int(h)).cuda(),
    torch.rand(1, 320, int(h), int(h)).cuda(),
    torch.rand(1, 320, int(h/2), int(h/2)).cuda(),
    torch.rand(1, 640, int(h/2), int(h/2)).cuda(),
    torch.rand(1, 640, int(h/2), int(h/2)).cuda(),
    torch.rand(1, 640, int(h/4),int(h/4)).cuda(),
    torch.rand(1, 1280, int(h/4), int(h/4)).cuda(),
    torch.rand(1, 1280, int(h/4), int(h/4)).cuda(),
    torch.rand(1, 1280, int(h/8), int(h/8)).cuda(),
    torch.rand(1, 1280, int(h/8), int(h/8)).cuda(),
    torch.rand(1, 1280, int(h/8), int(h/8)).cuda(),

]
x_vae_list=[
    torch.rand(1,512,int(HEIGHT/8),int(HEIGHT/8)).cuda(),
    torch.rand(1,512,int(HEIGHT/8),int(HEIGHT/8)).cuda(),
    torch.rand(1,256,int(HEIGHT/4),int(HEIGHT/4)).cuda(),
    torch.rand(1, 128, int(HEIGHT / 2), int(HEIGHT / 2)).cuda(),
    torch.rand(1, 128, HEIGHT, HEIGHT).cuda(),
]
y_vae_list=[
    torch.rand(1,512,int(HEIGHT/8),int(HEIGHT/8)).cuda(),
    torch.rand(1,512,int(HEIGHT/4),int(HEIGHT/4)).cuda(),
    torch.rand(1,512,int(HEIGHT/2),int(HEIGHT/2)).cuda(),
    torch.rand(1, 256, int(HEIGHT ), int(HEIGHT)).cuda(),
    torch.rand(1, 128, HEIGHT, HEIGHT).cuda(),
]
x_mul_list=[
    torch.rand(1, 128, int(HEIGHT), int(HEIGHT)).cuda(),
    torch.rand(1, 128, int(HEIGHT / 2), int(HEIGHT / 2)).cuda(),
    torch.rand(1, 256, int(HEIGHT / 4), int(HEIGHT / 4)).cuda(),
    torch.rand(1, 512, int(HEIGHT / 8), int(HEIGHT / 2)).cuda(),
    torch.rand(1, 512, int(HEIGHT / 8), int(HEIGHT / 2)).cuda()
]
text_feature=torch.rand(1,1,768).cuda()
z=torch.rand(1,8,int(h),int(h)).cuda()

t_list=[]
flop_list=[]
param_list=[]
model= SCBNet(4,True,0,('My_CrossAttnDownBlock2D', 'My_CrossAttnDownBlock2D', 'My_CrossAttnDownBlock2D', 'DownBlock2D'),False,(320,640,1280,1280),2,1,1,
              'silu',32,1e-05,768,8,False,None,None,False,'default',None,'rgb',
              (16,32,96,256),False).cuda()

t,f,p=analyse(model,(timestep,text_feature,cond_img,False))
t_list.append(t)
flop_list.append(f)
param_list.append(p)
del model
model=fusion_module().cuda()
t,f,p=analyse(model, (x_unet_list,x_unet_list))
t_list.append(t)
flop_list.append(f)
param_list.append(p)
del model
model= decoder().cuda()
t,f,p=analyse(model, (x_vae_list,y_vae_list))
t_list.append(t)
flop_list.append(f)
param_list.append(p)
del model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MultiScaleFeatureFusionModule()
model=model.to(device)
t,f,p=analyse(model,(x_mul_list,x_mul_list))
t_list.append(t)
flop_list.append(f)
param_list.append(p)
del model
print('==================ALL==================')
print("Time(ms):", numpy.sum(t_list))
print("FLOPs(G):", numpy.sum(flop_list))
print("Params(M):", numpy.sum(param_list))

