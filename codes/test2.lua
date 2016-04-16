require 'torch'
require 'image'
h=720
w=480
y=300.23
x=100.34
offset=15
im1 = image.gaussian{amplitude=5, 
                     normalize=true, 
                     width=w, 
                     height=h, 
                     sigma_horz=offset/w, 
                     sigma_vert=offset/h, 
                     mean_horz=x/w, 
                     mean_vert=y/h}
im1=image.scale(im1,120,80)
w=image.display(im1)
