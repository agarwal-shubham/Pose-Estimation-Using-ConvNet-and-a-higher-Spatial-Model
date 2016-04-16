require 'torch'
require 'image'
require 'optim'
require 'nn'--cpu
--require 'cunn'--cuda
------------------------------------------------------------------------------------------------------------------
									--CONVOLUTION LAYER to generate feature maps--
------------------------------------------------------------------------------------------------------------------

	-- input image dimensions
INfeat=1
height=240
width=360
ninputs=INfeat*width*height

	-- ConvNet layer dimensions
filtsize=5
poolsize=2
CNNstates={64,128,128}

	-- Note: the architecture of this ConvNet based on "http://cims.nyu.edu/~tompson/others/iclr2014_paper.pdf"
	-- (5x5 Conv -> ReLU ->  MaxPool(2x2)) * 2 + (5x5 Conv -> ReLU)
model = nn.Sequential()

   	-- stage 1:
model:add(nn.SpatialConvolutionMM(INfeat, CNNstates[1], filtsize, filtsize, 1, 1, (filtsize-1)/2, (filtsize-1)/2))
--SpatialConvolutionMM(infeat,outfeat,filtw,filth,stridew,strideh,padw,padh)
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
--SpatialMaxPooling(filtw,filth,stridew,strideh,padw,padh)
--width1  = op((widthprev  + 2*padw - filtw) / stridew + 1)
--height1 = op((heightprev + 2*padh - filth) / strideh + 1)
	
	-- stage 2:
model:add(nn.SpatialConvolutionMM(CNNstates[1], CNNstates[2], filtsize, filtsize, 1, 1, (filtsize-1)/2, (filtsize-1)/2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	
	--stage 3:
model:add(nn.SpatialConvolutionMM(CNNstates[2], CNNstates[3], filtsize, filtsize, 1, 1, (filtsize-1)/2, (filtsize-1)/2))
model:add(nn.ReLU())

------------------------------------------------------------------------------------------------------------------
									--FULLY CONNECTED LAYER to get class scores--
------------------------------------------------------------------------------------------------------------------

--[[	-- patch based architecture
FCstates={500,100,1}
model:add(nn.Reshape(CNNstates[3]*filtsize*filtsize))
	
	-- stage 1:
model:add(nn.Linear(CNNstates[3]*filtsize*filtsize, FCstates[1]))
model:add(nn.ReLU())	
	
	--stage 2:
model:add(nn.Linear(FCstates[1], FCstates[2]))
model:add(nn.ReLU())

	-- stage 3:
model:add(nn.Linear(FCstates[2], FCstates[1]))
model:add(nn.Sigmoid())
]]


	-- image based architecture
filtsize2 = 9
filtsize3 = 3
Nfeat = 3
CNNstates2={512,256,Nfeat}		-- 3 is the number of final class scores(no of joints)

   	-- stage 1:
model:add(nn.SpatialConvolutionMM(CNNstates[3], CNNstates2[1], filtsize2, filtsize2, 1, 1, (filtsize2-1)/2, (filtsize2-1)/2))
model:add(nn.ReLU())

	-- stage 2:
model:add(nn.SpatialConvolutionMM(CNNstates2[1], CNNstates2[2], filtsize3, filtsize3, 1, 1, (filtsize3-1)/2, (filtsize3-1)/2))
model:add(nn.ReLU())
	
	--stage 3:
model:add(nn.SpatialConvolutionMM(CNNstates2[2], CNNstates2[3], filtsize3, filtsize3, 1, 1, (filtsize3-1)/2, (filtsize3-1)/2))
model:add(nn.ReLU())

	-- moving it to GPU
--model = model:cuda()--cuda

------------------------------------------------------------------------------------------------------------------
										--MSE Criterion as ERROR FUNCTION--
------------------------------------------------------------------------------------------------------------------

criterion = nn.MSECriterion()
criterion.sizeAverage = true
--criterion = criterion:cuda()--cuda
x, dl_dx = model:getParameters()
data_dir = './../FLIC-full/images/'
feval = function(x_new)
	-- set x to x_new, if differnt
	-- the copy is really useless here though
	if x ~= x_new then
		x:copy(x_new)
	end

	-- select a new training sample
	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > #Paths then
		_nidx_ = 1
	end
	--_nidx_ = 1

	-- reset gradients because
	-- by default gradients are always accumulated, to accomodate batch methods
	dl_dx:zero()

	-- feed forward through your model and finally calculate the loss function(MSE)
	-- and it's derivative wrt x, for the sampled image
	
	model_out = model:forward(inputs)--cpu
	--model:forward(input:cuda())--cuda
	
	-- image.display(input)
	-- image.display(model.output)
	-- image.display(target)
	--layer=7			--layer=1,4,7,9,11,13
	--image.display{image = model:get(layer).weight,nrows=8}
	
	local loss_x = criterion:forward(model_out, targets)--cpu
	local dx_do = criterion:backward(model.output, targets)--cpu
	model:backward(inputs, dx_do)--cpu
	--local loss_x = (criterion:forward(model.output, target:cuda())):float()--cuda
	--model:backward(input:cuda(), criterion:backward(model.output, target:cuda()))--cuda
	--print(w5 - model:get(11).weight)


	-- return loss(x) and dloss/dx
	return loss_x, dl_dx
end

------------------------------------------------------------------------------------------------------------------
									--SGD TRAINING with NESTEROV MOMENTUM--
------------------------------------------------------------------------------------------------------------------

	-- Stochastic Gradient Descent with Nesterov momentum as back propagation procedure
	-- on a MSE error function
config = {
	learningRate = 0.1,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.2,
	nesterov = true,
	dampening = 0
}
state = config
-- load the dataset
dofile('./data.lua')
-- number of times to cycle over our training data

flag1 = 0
flag2 = 0
epochs = 5
batchSize = 1																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
for i = 1,epochs do
	loss = 0
	for j = 1,10,batchSize do
		--criterion = nn.MSECriterion()
		--trainer = nn.StochasticGradient(model, criterion)
		--trainer.learningRate = 0.01
		--process dataset
		--trainer:train(dataset)
----------------------------------------------------------------------------------------------------------------
------------------------------------------Batched data----------------------------------------------------------
----------------------------------------------------------------------------------------------------------------

		inputs = torch.Tensor(batchSize,1,240,360)
		targets = torch.Tensor(batchSize,3,60,90)

		for k=j,j+batchSize-1 do
			----------------------------------TRAINING IMAGE--------------------------
			local input = image.load(data_dir .. Paths[k])
			local ht = (#input)[2]
			local wd = (#input)[3]
			-- input -> 3x480x720
		    
		    --LCN on image
			input = image.rgb2y(input)
			ker = image.gaussian(5)
			m1 = nn.SpatialSubtractiveNormalization(1,ker)
			proc = m1:forward(input)
			m2 = nn.SpatialDivisiveNormalization(1,ker)
			processed = m2:forward(proc)

			-- resizing to save computation (orig-> 720x480)
			inputs[k-j+1] = image.scale(processed,360,240)
			-------------------------------------------------------------------------
			------------------------------TARGET IMAGE-------------------------------
			local target = torch.Tensor(3,60,90)		--3 heat maps
			
			local idx = joints[k][13][1]	--13 for nose
			local idy = joints[k][13][2]
			local offset = 10					--variance about joint location
			-- heatmap is a 2D Gaussian as a heat map with mean centered at joint location
			local heatmap0 = torch.Tensor(60,90):fill(0)
			if not idx~=idx then
				local heatmap = image.gaussian{amplitude=100, 
			                     normalize=false, 
			                     width=wd, 
			                     height=ht, 
			                     sigma_horz=offset/wd, 
			                     sigma_vert=offset/ht, 
			                     mean_horz=idx/wd, 
			                     mean_vert=idy/ht}
				-- target is smaller due to pooling
				heatmap = image.scale(heatmap,90,60)
				target[1] = heatmap
			else
				target[1] = heatmap0
			end

			idx = joints[k][1][1]	--1 for right shoulder
			idy = joints[k][1][2]	
			if not idx~=idx then
				heatmap = image.gaussian{amplitude=100,
			                     normalize=false, 
			                     width=wd, 
			                     height=ht, 
			                     sigma_horz=offset/wd, 
			                     sigma_vert=offset/ht, 
			                     mean_horz=idx/wd, 
			                     mean_vert=idy/ht}
				heatmap = image.scale(heatmap,90,60)
				target[2] = heatmap
			else
				target[2] = heatmap0
			end

			idx = joints[k][4][1]	--4 for left shoulder
			idy = joints[k][4][2]
			if not idx~=idx then
				heatmap = image.gaussian{amplitude=100, 
			                     normalize=false, 
			                     width=wd, 
			                     height=ht, 
			                     sigma_horz=offset/wd, 
			                     sigma_vert=offset/ht, 
			                     mean_horz=idx/wd, 
			                     mean_vert=idy/ht}
				heatmap = image.scale(heatmap,90,60)
				target[3] = heatmap
			else
				target[3] = heatmap0
			end
			targets[k-j+1] = target
		end
----------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------

		_,fs = optim.sgd(feval,x,config,state)
		
		print('loss = ' .. fs[1] .. ' for sample = '.. j .. ' in epoch = ' .. i)
		--loss = loss + fs[1]
		--[[
		]]
		if i>=3 then
			for k=1,batchSize do
				model_out[k][1][model_out[k][1]:lt(torch.max(model_out[k][1])*0.9)] = 0
				image.display{model_out[k][1], legend = 'output'}
				image.display{targets[k][1], legend = 'target'}
			end
		end
		--filt_disp=model:get(7).weight
		--image.display{image=filt_disp, legend='stage 3: weights'}
		
		collectgarbage()
	end
	--torch.save("learned_model.net",model)
	--torch.save("optim_state.t7",state)
	--local saved_model = torch.load("learned_model.net")
	--floss = loss / #Paths
	--print('loss = ' .. floss .. 'in epoch = ' .. i)
end
------------------------------------------------------------------------------------------------------------------
												--TESTING--
------------------------------------------------------------------------------------------------------------------

tloss=0
for i = 1,10 do
	
	local test_input = image.load(data_dir .. Paths[i])
	-- input -> 3x480x720
	--LCN on image
	test_input = image.rgb2y(test_input)
	ker = image.gaussian(5)
	m1 = nn.SpatialSubtractiveNormalization(1,ker)
	proc = m1:forward(test_input)
	m2 = nn.SpatialDivisiveNormalization(1,ker)
	processed = m2:forward(proc)

	local test_target = torch.Tensor(3,60,90)
	-- resizing to save computation (orig-> 720x480)
	test_input = image.scale(processed,360,240)
	-- target is smaller due to pooling

	local ht = (#test_input)[2]
	local wd = (#test_input)[3]
	local idx1 = joints[i][13][1]	--13 for nose
	local idy1 = joints[i][13][2]
	local idx2 = joints[i][1][1]	--1 for rsho
	local idy2 = joints[i][1][2]
	local idx3 = joints[i][4][1]	--4 for lsho
	local idy3 = joints[i][4][2]
	local offset = 10					--variance about joint location
	-- heatmap is a 2D Gaussian as a heat map with mean centered at joint location
	local heatmap = image.gaussian{amplitude=100, 
                     normalize=false, 
					width=wd, 
                     height=ht, 
                     sigma_horz=offset/wd, 
                     sigma_vert=offset/ht, 
                     mean_horz=idx1/wd, 
                     mean_vert=idy1/ht}
    heatmap = image.scale(heatmap,90,60)
	test_target[1] = heatmap
	heatmap = image.gaussian{amplitude=100, 
                     normalize=false, 
                     width=wd, 
                     height=ht, 
                     sigma_horz=offset/wd, 
                     sigma_vert=offset/ht, 
                     mean_horz=idx2/wd, 
                     mean_vert=idy2/ht}
    heatmap = image.scale(heatmap,90,60)
	test_target[2] = heatmap
	heatmap = image.gaussian{amplitude=100, 
                     normalize=false, 
                     width=wd, 
                     height=ht, 
                     sigma_horz=offset/wd, 
                     sigma_vert=offset/ht, 
                     mean_horz=idx3/wd, 
                     mean_vert=idy3/ht}
    heatmap = image.scale(heatmap,90,60)
	test_target[3] = heatmap
	

	local test_output = model:forward(test_input)--cpu
	local loss_x = criterion:forward(model.output, test_target)--cpu
	--local test_output = model:forward(test_input:cuda()):float()--cuda
	--local loss_x = (criterion:forward(model.output, target:cuda())):float()--cuda
	test_output[1][test_output[1]:lt(torch.max(test_output[1])*0.95)]=0
	image.display{test_output[1], legend = 'output'}
	image.display{test_target[1], legend = 'target'}
	print('loss = ' .. loss_x[1] .. ' in image = ' .. i)
		
	tloss = loss_x[1] + tloss
	collectgarbage()
end
finloss = tloss / 400
