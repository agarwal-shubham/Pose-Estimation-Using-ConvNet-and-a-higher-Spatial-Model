require 'torch'
require 'image'
--dofile('./data.lua')
require 'math'
require 'gm'

data_dir = './../FLIC-full/images/'

local input = image.load(data_dir .. Paths[1])
ht = (#input)[2]
wd = (#input)[3]
local offset = 10
--local idx = torch.Tensor(3,1)	--3 no of edges
--local idy = torch.Tensor(3,1)

finalPot = torch.Tensor(3,3,ht,wd)			-- 3 heat maps
finalPot:fill(0)
local c1=0
local c2=0
local c3=0
local relDiffy = 0
local relDiffx = 0

for i=1,500 do
	
	idx1 = joints[i][13][1]	--13 for nose
	idy1 = joints[i][13][2]
	idx2 = joints[i][4][1]	--4 for right shoulder
	idy2 = joints[i][4][2]
	idx3 = joints[i][1][1]	--1 for left shoulder
	idy3 = joints[i][1][2]
	if not idx2~=idx2 then
		relDiffy = idy2-idy1 + ((ht+1)/2)
		relDiffx = idx2-idx1 + ((wd+1)/2)
		heatmap1 = image.gaussian{amplitude=255, 
	                     normalize=false, 
	                     width=wd, 
	                     height=ht, 
	                     sigma_horz=offset/wd, 
	                     sigma_vert=offset/ht, 
	                     mean_horz=relDiffx/wd, 
	                     mean_vert=relDiffy/ht}
		finalPot[2][1]=finalPot[2][1]+heatmap1
		c1=c1+1
	end
	if not idx3~=idx3 then
		relDiffy = idy3-idy1 + ((ht+1)/2)
		relDiffx = idx3-idx1 + ((wd+1)/2)
		heatmap2 = image.gaussian{amplitude=255, 
	                     normalize=false, 
	                     width=wd, 
	                     height=ht, 
	                     sigma_horz=offset/wd, 
	                     sigma_vert=offset/ht, 
	                     mean_horz=relDiffx/wd, 
	                     mean_vert=relDiffy/ht}
		finalPot[3][1]=finalPot[3][1]+heatmap2
		c2=c2+1
	end
	if not idx3~=idx3 then
		if not idx2~=idx2 then
			relDiffy = idy3-idy2 + ((ht+1)/2)
			relDiffx = idx3-idx2 + ((wd+1)/2)
			heatmap3 = image.gaussian{amplitude=255, 
		                     normalize=false, 
		                     width=wd, 
		                     height=ht, 
		                     sigma_horz=offset/wd, 
		                     sigma_vert=offset/ht, 
		                     mean_horz=relDiffx/wd, 
		                     mean_vert=relDiffy/ht}
			finalPot[3][2]=finalPot[3][2]+heatmap3
			c3 = c3 + 1
		end
	end
end
finalPot[3][2]:div(c3)
finalPot[3][1]:div(c2)
finalPot[2][1]:div(c1)
heatmap0 = image.gaussian{amplitude=255, 
						normalize=false, 
						width=wd,
						height=ht,
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=1/2, 
						mean_vert=1/2}
finalPot[2][3] = image.rotate(finalPot[3][2],math.pi,'bilinear')	--rsho|lsho
finalPot[1][3] = image.rotate(finalPot[3][1],math.pi,'bilinear')	--nose|lsho
finalPot[1][2] = image.rotate(finalPot[2][1],math.pi,'bilinear')	--nose|rsho
finalPot[1][1] = heatmap0		--nose|nose
finalPot[2][2] = heatmap0		--rsho|rsho
finalPot[3][3] = heatmap0		--lsho|lsho
tempPot = torch.Tensor(3,3,60,90)
for i=1,3 do
	for j=1,3 do
		tempPot[i][j] = image.scale(finalPot[i][j],90,60)
	end
end
finalPot = tempPot

image.display(finalPot[1][2])
image.display(finalPot[3][1])
image.display(finalPot[2][3])

--[[
----------------------------------------------------------------------------------------------------------
										Experimenting
----------------------------------------------------------------------------------------------------------
heatmap_FN = image.gaussian{amplitude=255,
						normalize=true, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=(wd-100)/wd, 
						mean_vert=(ht-100)/ht}
heatmap1 = image.gaussian{amplitude=255, 
						normalize=true, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=idx1/wd, 
						mean_vert=idy1/ht}
heatmap2 = image.gaussian{amplitude=255, 
						normalize=true, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=idx2/wd, 
						mean_vert=idy2/ht}
heatmap3 = image.gaussian{amplitude=255, 
						normalize=true, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=idx3/wd, 
						mean_vert=idy3/ht}
heatmap_FN = image.scale(heatmap_FN,90,60)
heatmap1 = image.scale(heatmap1,90,60)
heatmap2 = image.scale(heatmap2,90,60)
heatmap3 = image.scale(heatmap3,90,60)
								-----------------------------------------------------
								-- experimenting
								res1 = image.convolve(heatmap1,finalPot[3][1],'full')	--face * lsho|face
								res2 = image.convolve(heatmap2,finalPot[3][2],'full')	--rsho * lsho|rsho
								inter = torch.cmul(res2,res1)
								inter = image.scale(inter,90,60)
								fin = torch.cmul(inter,heatmap3+heatmap_FN)
								image.display{image = res1, legend = 'face * lsho|face'}
								image.display{image = res2, legend = 'rsho * lsho|rsho'}
								image.display{image = heatmap3+heatmap_FN, legend = 'lsho(FN)'}
								image.display{image = fin, legend = 'result'}
----------------------------------------------------------------------------------------------------------
]]
