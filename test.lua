require 'helmholtz'

mytest = {}
 
tester = torch.Tester()

local function maxdiff(x,y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' then
      return d:abs():max()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():max()
   end
end
 

function mytest.TestInstantiate()
	local h = Helmholtz()
	tester:asserteq(h.nx, 1, "wrong default x dim")
	tester:asserteq(h.ny, 6, "wrong default y dim")
	tester:asserteq(h.nd, 9, "wrong default d dim")

	tester:asserteq(maxdiff(h.bG, torch.zeros(1, 1)), 0, "bG not zero")
	tester:asserteq(maxdiff(h.WG, torch.zeros(6, 2)), 0, "WG not zero")
	tester:asserteq(maxdiff(h.VG, torch.zeros(9, 7)), 0, "VG not zero")


end

function mytest.TestInitialize()
	tester:asserteq(h, nil, "h shouldn't exist")
end
 
tester:add(mytest)
tester:run()
