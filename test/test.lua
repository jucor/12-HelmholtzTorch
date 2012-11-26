package.loaded.hemlotz = nil
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

function mytest.TestWake()
	local h = Helmholtz()
  	local d = torch.zeros(1,9)
	h:Wake(d)
end
 
function mytest.TestFail()
	function failure()
		h.FunctionThatDoesNotExist()
	end
	assert(pcall(failure) == false)
end

tester:add(mytest)
tester:run()
