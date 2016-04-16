require 'image'
require 'nn'
require 'torch'
dofile('./data.lua')
dofile('./potential_priors.lua')
--[[
feval = function(x_new)
	--if x ~= x_new then
	--	x:copy(x_new)
	--end
	local nidx = (nidx or 0) + 1
	print(nidx)
	return nidx
end

for i=1,4 do
	local xn = feval
	print(xn)
end
]]
for i=1,1 do
	id=3101
	local img = image.load('./../FLIC-full/images/' .. Paths[id])
	local ht = (#img)[2]
	local wd = (#img)[3]
	local offset = 10
	local idx = torch.Tensor(3)
	local idy = torch.Tensor(3)
	local idxf = torch.Tensor(3)
	local idyf = torch.Tensor(3)
	idx[1] = joints[id][13][1]	--13 for nose
	idy[1] = joints[id][13][2]
	idx[2] = joints[id][4][1]	--4 for rsho
	idy[2] = joints[id][4][2]
	idx[3] = joints[id][1][1]	--1 for lsho
	idy[3] = joints[id][1][2]
	idxf[1] = joints[id][13][1]+185		-- false nose
	idyf[1] = joints[id][13][2]-15
	idxf[2] = joints[id][4][1]+227	--false rsho
	idyf[2] = joints[id][4][2]-16
	idxf[3] = joints[id][4][1]+290	--false lsho
	idyf[3] = joints[id][4][2]-21	
	heatmap = torch.Tensor(3,ht,wd)
	heatmap_FN = torch.Tensor(3,ht,wd)
	for j = 1,3 do
		heatmap[j] = image.gaussian{amplitude=255, 
						normalize=false, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=idx[j]/wd, 
						mean_vert=idy[j]/ht}
		heatmap_FN[j] = image.gaussian{amplitude=255,
						normalize=false, 
						width=wd, 
						height=ht, 
						sigma_horz=offset/wd, 
						sigma_vert=offset/ht, 
						mean_horz=idxf[j]/wd, 
						mean_vert=idyf[j]/ht}
	end
	dofile('./mrf.lua')
	--[[
	k=heatmap+heatmap_FN
	print(torch.max(k[1]))
	img[1][k[1]:gt(torch.max(k[1])-(0.04*255))]=1
	image.display(img)
	]]
	--[[
	image.display(img)
	image.display(heatmap_FN[2] + heatmap[2])
	image.display(heatmap_FN[1] + heatmap[1])
	image.display(heatmap_FN[3] + heatmap[3])
	]]
end
