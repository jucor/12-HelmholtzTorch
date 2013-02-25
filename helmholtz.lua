-- Implement the wake-sleep algorithm, as in [1]
-- [1] Kirby, K. G. A Tutorial on Helmholtz s. Department of Computer Science, Northern Kentucky Unviersity, June 2006. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.7019&rep=rep1&type=pdf.

require 'torch'
require 'image'
require 'dok'

function Sigmoid(x)
 --  return x:apply(function(x) return 1/(1+math.exp(-x)) end)
   local y = x:clone()
   y:mul(-1):exp():add(1):pow(-1)
   return y
end

function PassOneLayer(input, weights, backpropagate)
   local m
   if weights == nil then
      m = input:clone()
   else
      m = torch.mm(weights, input)
   end
   local x = Sigmoid(m)
   if not backpropagate then x:apply(torch.bernoulli) end
   return ExtendColumnByOne(x)
end

function ShrinkColumnByOne(x)
   assert(x:size(2) == 1, 'not a column vector')
   local y = x:clone()
   return y:resize(y:nElement()-1, 1)
end

function ExtendColumnByOne(x)
   assert(x:size(2) == 1, 'not a column vector')
   local y = x:clone()
   y:resize(y:nElement()+1, 1)
   y[y:nElement()] = 1
   return y
end

do
   local Helmholtz = torch.class('Helmholtz')

   function Helmholtz:__init(...)
      local _
      _, self.nx, self.ny, self.nd, self.stepB, self.stepW, self.stepV, self.backpropagate = dok.unpack(
         {...},
         'Constructor of a Helmholtz machine',
         'given a structure, returns an instance of the Helmholtz machine class',
         {arg='nx', type='number', help='number of neurons on top layer', default=1},
         {arg='ny', type='number', help='number of neurons on middle layer', default=6},
         {arg='nd', type='number', help='number of input in the data', default=9},
         {arg='stepB', type='number', help='stepsize for gradient on top layer', default=0.01},
         {arg='stepW', type='number', help='stepsize for gradient on middle layer', default=0.01},
         {arg='stepV', type='number', help='stepsize for gradient on bottom layer', default=0.15},
         {arg='backpropagate', type='boolean', help='if on, learning will use exact backpropagation instead of sampling intermediate neurons', default=false}
      )
      -- generative distribution
      self.bG = torch.zeros(self.nx, 1)
      self.WG = torch.zeros(self.ny, self.nx+1)
      self.VG = torch.zeros(self.nd, self.ny+1)
      -- recognition distribution
      self.WR = torch.zeros(self.nx, self.ny+1)
      self.VR = torch.zeros(self.ny, self.nd+1)
   end

   function Helmholtz:RecognizeExtended(d, backpropagate)
      if backpropagate == nil then backpropagate = false end
      -- Experience reality and  sense datum up through recognition network
      local d = ExtendColumnByOne(d)
      local y = PassOneLayer(d, self.VR, backpropagate)
      local x = PassOneLayer(y, self.WR, backpropagate)
      return y, x
   end

   function Helmholtz:GenerateExtended(backpropagate)
      if backpropagate == nil then backpropagate = false end
      -- Initiate a dream and pass signal down through generation network 
      local x = PassOneLayer(self.bG, nil, backpropagate)
      local y = PassOneLayer(x, self.WG, backpropagate)
      local d = PassOneLayer(y, self.VG, backpropagate)
      return d, y, x
   end

   function Helmholtz:Wake(d)
      -- Experience reality!
      local y, x = self:RecognizeExtended(d, self.backpropagate)

      -- Pass back down through generation network, saving computed probabilities
      local xi = PassOneLayer(self.bG, nil, true)
      local psi = PassOneLayer(x, self.WG, true)
      local delta = PassOneLayer(y, self.VG, true)

      -- Adjust generative weights by delta rule
      self.bG:add(self.stepB, x[{{1, self.nx}}] - xi[{{1, self.nx}}])
      self.WG:addmm(self.stepW, (y[{{1, self.ny}}] - psi[{{1, self.ny}}]), x:t())
      self.VG:addmm(self.stepV, (d[{{1, self.nd}}] - delta[{{1, self.nd}}]), y:t())
   end

   function Helmholtz:Sleep()
      -- Sample a dream
      local d, y, x = self:GenerateExtended(self.backpropagate)

      -- Pass back up through recognition network, saving computed probabilities
      local psi = PassOneLayer(d, self.VR, true)
      local xi = PassOneLayer(y, self.WR, true)

      -- Adjust generative weights by delta rule
      self.VR:addmm(self.stepV, (y[{{1, self.ny}}] - psi[{{1, self.ny}}]), d:t())
      self.WR:addmm(self.stepW, (x[{{1, self.nx}}] - xi[{{1, self.nx}}]), y:t())
   end

   function Helmholtz:Sample()
      local d = self:GenerateExtended()
      return ShrinkColumnByOne(d)
   end

end
