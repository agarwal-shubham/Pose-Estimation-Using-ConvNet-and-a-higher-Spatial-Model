require 'torch'
require 'image'
require 'nn'

local filePath = './data_paths(copy).csv'
local jointlocn = './joints_data(copy).csv'
local csvFile = io.open(filePath, 'r')

-- Split string
function string:split(sep)
	local sep, fields = sep, {}
	local pattern = string.format("([^%s]+)", sep)
	self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
	return fields
end

local i = 0
Paths={}
-- Note: if you use 'paths' as varialble name image.display() will not work(unknown reasons) 

-- inputs = torch.Tensor(5000,90,60)
for line in csvFile:lines('*l') do
	i = i + 1
	local l = line:split(',')
	local stri = ''
	for key, val in ipairs(l) do
		stri = stri .. val
	end
	Paths[i]=stri
	--				Storing data (memory problems)
	--[[
	local img = image.load('/home/shubham/Desktop/SMAI Project/FLIC-full/images/' .. Paths[i])
	img=image.rgb2y(img)
	--						LCN
	ker = image.gaussian(9)
	m1 = nn.SpatialSubtractiveNormalization(1,ker)
	proc = m1:forward(img)
	m2 = nn.SpatialDivisiveNormalization(1,ker)
	processed = m2:forward(proc)
	--				cropping for computation
	img=image.scale(img,90,60)
	inputs[i]=img
	]]
end
csvFile:close()
local csvFile = io.open(jointlocn, 'r')

i = 0
joints = torch.Tensor(5000,13,2)
for line in csvFile:lines('*l') do
	i = i + 1
	local l = line:split(',')
	for key, val in ipairs(l) do
	  	--if val~=val then
	  	--	val = -1
	  	--end
	  	
	  	--[[
	  			Joint keying/indexing
	  			[1]  -> lsho
	  			[2]  -> lelb
	  			[3]  -> lwri
	  			[4]  -> rsho
	  			[5]  -> relb
	  			[6]  -> rwri
	  			[7]  -> lhip
	  			[8]  -> lkne
	  			[9]	 -> lank
	  			[10] -> rhip
	  			[11] -> rkne
	  			[12] -> rank
	  			[13] -> nose
	  			
	  	]]
		if key< 13 then
			joints[math.floor(((i-1)/2)+1)][key][((i+1)%2)+1]=val
		elseif key==17 then
			joints[math.floor(((i-1)/2)+1)][13][((i+1)%2)+1]=val
		end
	end
end
csvFile:close()

--[[
local x=500
print(#Paths)
print(joints[x][3][1])
print(joints[x][3][2])
print(Paths[x])
local img = image.load('/home/shubham/Desktop/SMAI Project/FLIC-full/images/' .. Paths[x])
-- local img = image.load('/home/shubham/Desktop/SMAI Project/FLIC-full/images/12-oclock-high-special-edition-00098731.jpg')
print(#img)
print(img[1][300][350])
image.display(img)
]]
