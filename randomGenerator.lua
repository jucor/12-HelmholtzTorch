-- Computes the cdf of a discrete distribution, normalizing it as needed
function ComputeCDF(p)
   local pnorm = p:resize(p:nElement(),1):cumsum(1)
   local totalmass = pnorm[-1][1]
   if totalmass < 1e-16 then error('Not enough positive probabilities') end
   return pnorm:resizeAs(p) / totalmass
end

-- Sample N iid from a distribution on {1,..,#p}, crudely, O(N #p) complexity
function RandomInteger(p, N)
   N = N or 1
   local cdf = ComputeCDF(p)
   local uniform = torch.rand(N)
   return uniform:apply(function(x) return cdf:lt(x):sum()+1 end)
end


