-- Implement the wake-sleep algorithm, as in [1]
-- [1] Kirby, K. G. A Tutorial on Helmholtz s. Department of Computer Science, Northern Kentucky Unviersity, June 2006. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.7019&rep=rep1&type=pdf.

require 'torch'
require 'image'
require 'dok'

do
   function Sigmoid(x)
    --  return x:apply(function(x) return 1/(1+math.exp(-x)) end)
      x:mul(-1):exp():add(1):pow(-1)
      return x
   end

   function ShrinkColumnByOne(x)
      assert(x:size(2) == 1, 'not a column vector')
      x:resize(x:nElement()-1,1)
      return x
   end

   function ExtendColumnByOne(x)
      assert(x:size(2) == 1, 'not a column vector')
      x:resize(x:nElement()+1,1)
      x[x:nElement()] = 1
      return x
   end

   local Helmholtz = torch.class('Helmholtz')
   function Helmholtz:__init(...)
      _, self.nx, self.ny, self.nd, self.stepB, self.stepW, self.stepV, self.exact = dok.unpack(
         {...},
         'Constructor of a Helmholtz machine',
         'given a structure, returns an instance of the Helmholtz machine class',
         {arg='nx',type='number',help='number of neurons on top layer', default=1},
         {arg='ny',type='number',help='number of neurons on middle layer', default=6},
         {arg='nd',type='number',help='number of input in the data', default=9},
         {arg='stepB',type='number',help='stepsize for gradient on top layer', default=0.01},
         {arg='stepW',type='number',help='stepsize for gradient on middle layer', default=0.01},
         {arg='stepV',type='number',help='stepsize for gradient on bottom layer', default=0.15},
         {arg='exact',type='boolean',help='if on, learning will use exact backpropagation instead of sampling intermediate neurons', default=false}
      )
      -- generative distribution
      self.bG = torch.zeros(self.nx, 1)
      self.WG = torch.zeros(self.ny, self.nx+1)
      self.VG = torch.zeros(self.nd, self.ny+1)
      -- recognition distribution
      self.WR = torch.zeros(self.nx, self.ny+1)
      self.VR = torch.zeros(self.ny, self.nd+1)
   end

   function Helmholtz:Wake(d )
      -- Experience reality!
      local d = ExtendColumnByOne(d)
      -- Pass sense datum up through recognition network
      local y = ExtendColumnByOne(
                Sigmoid(torch.mm(self.VR, d)):apply(torch.bernoulli)
            )
      local x = ExtendColumnByOne(
               Sigmoid(torch.mm(self.WR, y)):apply(torch.bernoulli)
            )

      -- Pass back down through generation network, saving computer probabilities
      local xi = Sigmoid(self.bG:clone())
      local psi = Sigmoid(torch.mm(self.WG, x))
      local delta = Sigmoid(torch.mm(self.VG, y))

      -- Adjust generative weights by delta rule 
      self.bG:add(self.stepB, x[{{1,self.nx}}] - xi)
      self.WG:addmm(self.stepW, (y[{{1,self.ny}}] - psi), x:t())
      self.VG:addmm(self.stepV, (d[{{1,self.nd}}] - delta), y:t())
   end

   function Helmholtz:SampleExtended()
      -- Initiate a dream!
      local x = ExtendColumnByOne(
               Sigmoid(self.bG:clone()):apply(torch.bernoulli)
            )

      -- Pass dream signal down through generation network 
      local y = ExtendColumnByOne(
                Sigmoid(torch.mm(self.WG, x)):apply(torch.bernoulli)
            )
      local d = ExtendColumnByOne(
               Sigmoid(torch.mm(self.VG, y)):apply(torch.bernoulli)
            )

      return d, y, x
   end

   function Helmholtz:Sample()
      d, y, x = self:SampleExtended()
      return ShrinkColumnByOne(d), ShrinkColumnByOne(y), ShrinkColumnByOne(x)
   end

   function Helmholtz:Sleep()
      local d, y, x = self:SampleExtended()

      -- Pass back up through recognition network, saving computer probabilities
      local psi = Sigmoid(torch.mm(self.VR, d))
      local xi = Sigmoid(torch.mm(self.WR, y))

      -- Adjust generative weights by delta rule 
      self.VR:addmm(self.stepV, (y[{{1,self.ny}}] - psi), d:t())
      self.WR:addmm(self.stepW, (x[{{1,self.nx}}] - xi), y:t())
   end
end
