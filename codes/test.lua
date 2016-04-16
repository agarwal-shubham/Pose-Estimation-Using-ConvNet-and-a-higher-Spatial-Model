require 'torch'
require 'image'

local jointlocn = '/home/shubham/Desktop/SMAI Project/codes/joints_data(copy).csv'

-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


local csvFile = io.open(jointlocn, 'r')  
local header = csvFile:read()

local i = 0
joints = torch.Tensor(5000,13,2)
for line in csvFile:lines('*l') do  
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
  	--if val~=val then
  	--	val = -1
  	--end	
  	if key< 13 then
  		joints[math.floor(((i-1)/2)+1)][key][((i+1)%2)+1]=val
  	elseif key==17 then
  		joints[math.floor(((i-1)/2)+1)][13][((i+1)%2)+1]=val
  	end
  end
end
print(joints[2])
csvFile:close()
