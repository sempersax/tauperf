import math


class andrew_variables:
    """A class to compute Andrews variables"""
    def __init__(self,HADEnergy,EMEnergy,numTotCells):
        self.HADEnergy   = HADEnergy
        self.EMEnergy    = EMEnergy
        self.numTotCells = numTotCells
        
        self.HADtoEMEnergy,self.EnergyTonCells = self.getVariables()


    def getVariables(self):
        energy_grn = 0.1
        energy_max = 1000000000.
        nCells_grn = 0.1
        
        HADtoEMenergy  = -9999.
        EnergyTonCells = -9999.
        
            
        if abs(self.EMEnergy)<= energy_grn:
            if abs(self.HADEnergy)<=energy_grn:
                if (self.EMEnergy>=0. and self.HADEnergy>=0.) or (self.EMEnergy<0. and self.HADEnergy<0.):
                    HADtoEMenergy = 1.
                else:
                    HADtoEMenergy = -1.
            elif abs(self.HADEnergy)<energy_max:
                if self.EMEnergy>=0.: HADtoEMenergy = self.HADEnergy / energy_grn
                else: HADtoEMenergy = - self.HADEnergy / energy_grn
            else:
                if (self.EMEnergy>=0. and self.HADEnergy>=0.) or (self.EMEnergy<0. and self.HADEnergy<0.):
                    HADtoEMenergy = energy_max / energy_grn
                else:
                    HADtoEMenergy = - energy_max / energy_grn
        elif self.HADEnergy>=energy_max: HADtoEMenergy = energy_max/self.EMEnergy
        elif self.HADEnergy<=-energy_max: HADtoEMenergy = -energy_max/self.EMEnergy
        else: HADtoEMenergy = self.HADEnergy/self.EMEnergy
                  
              
        EMHADenergy = self.EMEnergy + self.HADEnergy
        if self.numTotCells == 0:
            if abs( EMHADenergy ) <= energy_grn:
                if EMHADenergy >= 0.: EnergyTonCells = 1.
                else: EnergyTonCells = -1
            elif ( fabs ( EMHADenergy ) <= energy_max ):
                EnergyTonCells = EMHADenergy / nCells_grn
            else:
                if EMHADenergy >= 0.: EnergyTonCells = energy_max / nCells_grn ;
                else: EnergyTonCells = - energy_max / nCells_grn
        elif EMHADenergy >= energy_max: EnergyTonCells = energy_max /( float(self.numTotCells) )
        elif EMHADenergy <= - energy_max: EnergyTonCells = - energy_max /( float(self.numTotCells) )
        else: EnergyTonCells = EMHADenergy /( float(self.numTotCells) )


        return HADtoEMenergy,EnergyTonCells
