require 'randomGenerator'

randomtest = {}
tester = torch.Tester()

function tester:assertError(f, message)
   status, err = pcall(f)
   self:assert_sub(status == false, string.format('%s\n%s  condition=%s',message,' ERROR violation ', 'should have errored'))
end

function randomtest.TestAssertError()
   tester:assertError(function() error('hello') end, 'Error not caught')
end

function randomtest.TestCDF()
   local p = {
      [{1, 0, 0}] = {1, 1, 1},
      [{0, 1, 0}] = {0, 1, 1},
      [{0, 0, 1}] = {0, 0, 1},
      [{.1, .1, .2}] = {.25, .5, 1}
      }
   for k,v in pairs(p) do
      tk = torch.Tensor({k})
      tv = torch.Tensor({v})
      tester:assertTensorEq(CumsumAndNormalize(tk), tv, 1e-16, 'did not normalize correctly') 
   end
   tester:assertError(function() CumsumAndNormalize(torch.Tensor({0, 0, 0})) end, 'accepted vector of 0s')
end

tester:add(randomtest)
tester:run()
