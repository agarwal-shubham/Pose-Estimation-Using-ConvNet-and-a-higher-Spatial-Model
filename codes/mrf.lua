require 'torch'
require 'image'
--dofile('/home/shubham/Desktop/SMAI Project/codes/data.lua')
--dofile('/home/shubham/Desktop/SMAI Project/codes/potential_priors.lua')
--dofile('/home/shubham/Desktop/SMAI Project/codes/test3.lua')
require 'math'
--require 'gm'

data_dir = './../FLIC-full/images/'

input_sample = 	torch.Tensor(3,60,90)	--replace with resulting vector from convolution layer
--input_sample[1] = image.scale((heatmap[1] + heatmap_FN[1]),90,60)		--nose + false neg.
test_output[1][test_output[1]:lt(torch.max(test_output[1])*0.96)] = 0;
test_output[2][test_output[2]:lt(torch.max(test_output[2])*0.96)] = 0;
test_output[3][test_output[3]:lt(torch.max(test_output[3])*0.96)] = 0;
input_sample[1] = test_output[1]:double() 		--nose + false neg.
input_sample[2] = test_output[2]:double()		--rsho + flase neg.
input_sample[3] = test_output[3]:double()		--lsho + flase neg.

res = torch.Tensor(3,60,90)
res:fill(1)
nJoints = 3

for i=1,nJoints do
	for j=1,nJoints do
		--if i==j then
		--	res[i] = torch.cmul(res[i],input_sample[j])
		--else
			tempo = image.convolve(input_sample[j],finalPot[i][j],'same')
			--image.display(tempo)
			--print(#tempo)
			--60x90 conv 60x90 = 119x179(for 'full'), 1x1(for 'valid'), 60x90(for 'same')
			res[i]=torch.cmul(res[i],tempo)
			--image.display(res[i])
		--end
	end
end
x=wd
y=ht
--[[
image.display{image = image.scale(input_sample[2],x,y), legend = 'rsho(FN)'}
image.display{image = image.scale(res[2],x,y), legend = 'new rsho'}
image.display{image = image.scale(input_sample[1],x,y), legend = 'nose(FN)'}
image.display{image = image.scale(res[1],x,y), legend = 'new nose'}
image.display{image = image.scale(input_sample[3],x,y), legend = 'lsho(FN)'}
image.display{image = image.scale(res[3],x,y), legend = 'new lsho'}
]]
local im = image.load('./../FLIC-full/images/' .. Paths[id])	--id is taken from test3.luan file
local imCopy = image.load('./../FLIC-full/images/' .. Paths[id])

--k=heatmap+heatmap_FN
k = image.scale(input_sample,x,y)
imCopy[1][k[1]:gt(torch.max(k[1])*0.96)]=1.3
imCopy[2][k[2]:gt(torch.max(k[2])*0.96)]=1.3
imCopy[3][k[3]:gt(torch.max(k[3])*0.96)]=1.3
image.display(imCopy)
image.save('mrf_results1.jpg',imCopy)
resUpscaled = torch.Tensor(3,ht,wd)
resUpscaled[1] = image.scale(res[1],x,y)
resUpscaled[2] = image.scale(res[2],x,y)
resUpscaled[3] = image.scale(res[3],x,y)

im[1][resUpscaled[1]:gt(torch.max(resUpscaled[1])*0.96)]=1.3
im[2][resUpscaled[2]:gt(torch.max(resUpscaled[2])*0.96)]=1.3
im[3][resUpscaled[3]:gt(torch.max(resUpscaled[3])*0.96)]=1.3
image.display(im)
image.save('mrf_results2.jpg',im)

--[[
numNodes = 3
adjacency = gm.adjacency.full(numNodes)

numStates = 2
gr = gm.graph{adjacency=adjacency, nStates=numStates, maxIter=10, type='mrf', verbose=true}

print(adjacency)

-- unary potentials (fed from the ConvNet layer)
nodePot = {{1,2},{3,4},{5,6}}

-- joint potentials (learned from the training data)
jointPot = torch.Tensor(gr.nEdges,numStates,numStates)
basic = torch.Tensor{{0,9},{9,0}}
for e=1,gr.nEdges do
	jointPot[e] = basic
end

-- set potentials
gr:setPotentials(nodePot,jointPot)

opt = gr:decode('bp')
print(opt)
]]
