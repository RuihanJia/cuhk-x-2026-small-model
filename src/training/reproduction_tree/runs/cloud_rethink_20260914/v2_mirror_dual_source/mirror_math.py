import numpy as np
def horizontal_mirror(x):
 assert x.shape[-1]==224 and x.shape[-2]==224
 return x.flip(-1)
def aggregate(z,kind):
 z=np.asarray(z,dtype=np.float32)
 assert z.shape[-2:]==((7,40)if kind=='Base'else(5,40))
 return ((.5*z[...,:5,:].mean(-2)+.25*z[...,5,:]+.25*z[...,6,:])if kind=='Base'else z.mean(-2)).astype(np.float32)
def mix_original(original,mirrored):
 assert original.dtype==mirrored.dtype==np.float32 and original.shape==mirrored.shape
 return (.5*original+.5*mirrored).astype(np.float32)
