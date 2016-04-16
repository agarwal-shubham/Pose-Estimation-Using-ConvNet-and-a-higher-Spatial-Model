require 'image'
require 'nn'
lena = image.rgb2y(image.lena())
ker = image.gaussian(9)
m1=nn.SpatialSubtractiveNormalization(1,ker)
proc = m1:forward(lena)
m2=nn.SpatialDivisiveNormalization(1,ker)
processed = m2:forward(proc)
w1=image.display(lena)
w2=image.display(proc)
x1 = 230
x2 = 294
y1 = 230
y2 = 294
fin = image.crop(proc, x1, y1, x2, y2)
w3=image.display(fin)
print(fin:size())