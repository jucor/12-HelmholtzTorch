function CumsumAndNormalize(p)
   local pnorm = p:resize(p:nElement(),1):cumsum(1)
   local renorm = pnorm[-1][1]
   if renorm < 1e-16 then error('Not enough positive probabilities') end
   return pnorm:resizeAs(p) / renorm
end

function RandomInteger(p, N)
   N = N or 1
   local cdf = CumsumAndNormalize(p)
   return apply(torch.rand(N), function(x) cdf:lt(x):sum() end)
end


