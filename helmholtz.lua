-- Implement the wake-sleep algorithm, as in [1]
-- [1] Kirby, K. G. A Tutorial on Helmholtz s. Department of Computer Science, Northern Kentucky Unviersity, June 2006. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.7019&rep=rep1&type=pdf.


require 'torch'
require 'dok'

do 
   local Helmholtz = torch.class('Helmholtz')

   function Helmholtz:__init()
      -- dimensions of the layers
      self.nx = 1
      self.ny = 6
      self.nd = 9
      -- generative distribution
      self.bG = torch.zeros(self.nx, 1)
      self.WG = torch.zeros(self.ny, self.nx+1)
      self.VG = torch.zeros(self.nd, self.ny+1)
      -- recognition distribution
      self.WR = torch.zeros(self.nx, self.ny+1)
      self.VR = torch.zeros(self.ny, self.nd+1)
   end

	function Helmholtz:Wake(data)
	 	 
	end

end
