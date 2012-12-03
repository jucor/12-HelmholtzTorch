require 'helmholtz'

function HashImage(d)
   local hash = 0
   for i=1,3 do
      for j=1,3 do
         hash = hash*2 + d[i][j]
      end
   end
   return hash
end

function CountFreqs(d)
   local freqs = {}
   for i=1,d:size(1) do
      local h = HashImage(d[i])
      if freqs[h] == nil then
         freqs[h] = 1
      else
         freqs[h] = freqs[h] + 1
      end
   end
   return freqs
end

function SampleKirby()
   local d = torch.zeros(3,3)
   -- flip one column chosen with proba 1/3
   local col = torch.random(1,3)
   d[{{},col}] = 1
   -- transpose to horizontal with proba 1/3
   if torch.rand(1)[1] < .3 then d:t() end
   -- flip to white on black with proba 1/2 
   if torch.rand(1)[1] < .5 then d:apply(function(x) return 1 - x end) end
   return d
end

function DemoKirby()
   local N = 1000
   local d = torch.zeros(N,3,3)
   for i=1,N do
      d[i] = SampleKirby()
   end
   image.display{image=d[{{1,16},{},{}}],zoom=30,padding=0}

   local f = CountFreqs(d)
   print(f)
end

