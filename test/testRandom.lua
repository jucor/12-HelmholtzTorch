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
      tester:assertTensorEq(ComputeCDF(tk), tv, 1e-16, 'did not normalize correctly') 
   end
   -- call with a null vector should return error
   tester:assertError(function() ComputeCDF(torch.Tensor({0, 0, 0})) end, 'accepted vector of 0s')
end

function randomtest.TestSampling()
   local p = {
      [{1, 0, 0}] = 1,
      [{0, 1, 0}] = 2,
      [{0, 0, 1}] = 3
      }
   local N = 10
   for k,v in pairs(p) do
      tk = torch.Tensor({k})
      tv = torch.Tensor(N):fill(v)
      tester:assertTensorEq(RandomInteger(tk), torch.Tensor({v}), 1e-16, 'did not sample correctly for a dirac in ' .. v) 
      tester:assertTensorEq(RandomInteger(tk, N), tv, 1e-16, 'did not sample correctly for ' .. N .. ' iid dirac in ' .. v ) 
   end
end
   

tester:add(randomtest)
tester:run()
