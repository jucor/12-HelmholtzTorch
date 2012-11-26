-- This script shows that tests are not run in independent environments: they share the same global --
require 'torch'

tester = torch.Tester()
mytest = {}

function mytest.FirstTest()
   tester:assert(a == nil, 'a should not exist')
   tester:assert(b == nil, 'b should not exist')
   a = 2
end

function mytest.SecondTest()
   tester:assert(a == nil, 'a should not exist')
   tester:assert(b == nil, 'b should not exist')
   b = 2
end

tester:add(mytest)
tester:run()

