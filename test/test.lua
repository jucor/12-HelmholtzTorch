require 'helmholtz'

mytest = {}
tester = torch.Tester()

function mytest.TestInstantiate()
   local h = Helmholtz()
   tester:asserteq(h.nx, 1, "wrong default x dim")
   tester:asserteq(h.ny, 6, "wrong default y dim")
   tester:asserteq(h.nd, 9, "wrong default d dim")

   tester:assertTensorEq(h.bG, torch.zeros(1, 1), 1e-16, "bG not zero")
   tester:assertTensorEq(h.WG, torch.zeros(6, 2), 1e-16, "WG not zero")
   tester:assertTensorEq(h.VG, torch.zeros(9, 7), 1e-16, "VG not zero")
end

function mytest.TestInitialize()
   tester:asserteq(h, nil, "h shouldn't exist")
end

function mytest.TestSigmoid()
   local v = {-1, 0, 1}
   local x = torch.Tensor(v)
   local y = Sigmoid(x)
   tester:asserteq(y, x, 'did not return the tensor')
   for i=1,#v do
      tester:assertne(x[i], v[i],'did not modify tensor')
      tester:asserteq(x[i], 1/(1+math.exp(-v[i])),'did not compute a sigmoid')
   end
end

function mytest.TestShrinkColumnByOne()
   local v = {10, 11, 12}
   local x = torch.Tensor({v}):t()
   tester:asserteq(x:nElement(), 3, 'wrong size before')
   tester:asserteq(x:size(1), 3, 'wrong number of rows')
   tester:asserteq(x:size(2), 1, 'wrong number of columns')
   local y = ShrinkColumnByOne(x)
   tester:asserteq(y, x, 'did not return the tensor')
   tester:asserteq(x:nElement(), 2, 'wrong size after')
   tester:asserteq(x[{1,1}], 10, 'not starting by 10')
   tester:asserteq(x[2][1], 11, 'not ending by 11')
end 

function mytest.TestExtendColumnByOne()
   local v = {10, 11, 12}
   local x = torch.Tensor({v}):t()
   tester:asserteq(x:nElement(), 3, 'wrong size before')
   tester:asserteq(x:size(1), 3, 'wrong number of rows')
   tester:asserteq(x:size(2), 1, 'wrong number of columns')
   local y = ExtendColumnByOne(x)
   tester:asserteq(y, x, 'did not return the tensor')
   tester:asserteq(x:nElement(), 4, 'wrong size after')
   tester:asserteq(x[{4,1}], 1, 'not ending by 1')
   tester:asserteq(x[4][1], 1, 'not ending by 1')
end

function mytest.TestSample()
   local h = Helmholtz()
   local d, y, x = h:Sample()
   tester:asserteq(x:nElement(), 1, 'wrong size')
   tester:asserteq(y:nElement(), 6, 'wrong size')
   tester:asserteq(d:nElement(), 9, 'wrong size')
   d, y, x = h:SampleExtended()
   tester:asserteq(x:nElement(), 2, 'wrong size')
   tester:asserteq(y:nElement(), 7, 'wrong size')
   tester:asserteq(d:nElement(), 10, 'wrong size')
end

function mytest.TestWake()
   local h = Helmholtz()
   local d = torch.zeros(9,1)
   local oldBG = h.bG:clone()
   local oldVG = h.VG:clone()
   local oldWG = h.WG:clone()
   h:Wake(d)
   tester:assertge((oldBG-h.bG):abs():max(), 1e-16, "h.bG has not changed") 
   tester:assertge((oldVG-h.VG):abs():max(), 1e-16, "h.VG has not changed") 
   tester:assertge((oldWG-h.WG):abs():max(), 1e-16, "h.WG has not changed") 
end

function mytest.TestSleep()
   local h = Helmholtz()
   local oldVR = h.VR:clone()
   local oldWR = h.WR:clone()
   h:Sleep()
   tester:assertge((oldVR-h.VR):abs():max(), 1e-16, "h.VR has not changed") 
   tester:assertge((oldWR-h.WR):abs():max(), 1e-16, "h.WR has not changed") 
end
 
function mytest.TestFail()
   local function failure()
      h.FunctionThatDoesNotExist()
   end
   tester:assert(pcall(failure) == false, 'pcall to nonexisting function should have failed')
end

tester:add(mytest)
tester:run()
