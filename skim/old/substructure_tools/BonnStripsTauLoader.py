# --> StripsTauLoader class
from ROOT import *
import math
from EventVariables import *

gROOT.LoadMacro("load_stl_float.h+")
gROOT.LoadMacro("load_stl_int.h+")


class BonnStripsTauLoader:
    """A class to compute and dump tau variables related to the strips layer"""

    #-----------------------------------------------------------
    def __init__(self, tree, reco_index):
        """Constructor"""

        self._tree           = tree
        self._recoIndex      = reco_index
        self.reco4Vector     = self.getReco4Vector()
        self.pi0Bonn_TauShot_shots = self.getPi0BonnShots(0.2) # --> DeltaR cut 
        self.pi0Bonn_TauShot_nShot = len(self.pi0Bonn_TauShot_shots)
        print self.pi0Bonn_TauShot_nShot,self._tree.tau_pi0Bonn_TauShot_nShot [self._recoIndex]
        self.nEffGammaClusters     = self.getNEffGammaClusters()
        self.StripMass  = self.getInvarMass()
        self.EffStripMass = self.getInvarMass(int(ceil(self.nEffGammaClusters)))
            
        self.Efrac_tot          = {}
        self.closeTrackDeltaR   = {}
        self.closeTrackDeltaPhi = {}
        self.closeTrackDeltaEta = {}
        self.closeTrackEfrac    = {}
        for ishot in range(0,self.pi0Bonn_TauShot_nShot):
            self.Efrac_tot[ishot] = self.getEfrac_tot(ishot)
            track_index = self.getIndexOfClosestTrackToShot(ishot)
            if track_index>-1:
                self.closeTrackDeltaR[ishot],self.closeTrackDeltaEta[ishot],self.closeTrackDeltaPhi[ishot],self.closeTrackEfrac[ishot]=self.getClosestTrackToShot_Variable(ishot,track_index)
            else:
                self.closeTrackDeltaR   [ishot] = 0
                self.closeTrackDeltaPhi [ishot] = 0
                self.closeTrackDeltaEta [ishot] = 0
                self.closeTrackEfrac    [ishot] = 0
                
        self.DeltaR_leadclusters = 0
        self.DeltaEta_leadclusters = 0
        self.DeltaPhi_leadclusters = 0
        if self.pi0Bonn_TauShot_nShot>1:             
            self.DeltaR_leadclusters,self.DeltaEta_leadclusters,self.DeltaPhi_leadclusters= self.getLeadingSubleadingShots_Variable()

        return None



    #-----------------------------------------------------------
    def getReco4Vector(self):
        """Get the TLorentzVector for the reco. tau"""
        pt  = self._tree.tau_pt[self._recoIndex]
        eta = self._tree.tau_eta[self._recoIndex]
        phi = self._tree.tau_phi[self._recoIndex]

        vector = TLorentzVector()
        vector.SetPtEtaPhiM(pt, eta, phi, 0)

        return vector
    #------------------------------------------------------------
    def getPi0BonnShots(self,deltaR_cut):
        """Return the shots found by Bonn shot finding alg"""
        n = self._tree.tau_pi0Bonn_TauShot_nShot[self._recoIndex]
        shots = []
        for i in range(0,n):
            pt  = self._tree.tau_pi0Bonn_TauShot_Shot_pt[self._recoIndex][i]
            eta = self._tree.tau_pi0Bonn_TauShot_Shot_eta[self._recoIndex][i]
            phi = self._tree.tau_pi0Bonn_TauShot_Shot_phi[self._recoIndex][i]
            vector = TLorentzVector()
            vector.SetPtEtaPhiM(pt,eta,phi,0)
            deltaR = vector.DeltaR( self.reco4Vector)
            if deltaR<deltaR_cut:
                shots.append(vector)
        return shots

    #-----------------------------------------------------------
    def getNEffGammaClusters(self):
        """Get number of effective gamma candidates"""
        E_num  = 0
        E_den = 0

        for ishot in range(0,self.pi0Bonn_TauShot_nShot):
            E_num  += self.pi0Bonn_TauShot_shots[ishot].E()
            E_den += E_num**2

        if E_num>0: return (E_num**2)*(E_den**(-1))
        else: return 0.


    #-----------------------------------------------------------
    def getIndexOfClosestTrackToShot(self,shot_index):
        """Find the closest track to the shot with a DeltaR requirement""" 
        DeltaR = 100000
        index_track = -1
        eta_shot = self._tree.tau_pi0Bonn_TauShot_Shot_eta[self._recoIndex][shot_index]
        phi_shot = self._tree.tau_pi0Bonn_TauShot_Shot_phi[self._recoIndex][shot_index]

        # --> Loop over the tracks
        for itrack in range(0, self._tree.tau_track_n[self._recoIndex]):
            eta_track = self._tree.tau_track_atTJVA_eta[self._recoIndex][itrack]
            phi_track = self._tree.tau_track_atTJVA_phi[self._recoIndex][itrack]
            newDeltaR = rawDeltaR(eta_shot,eta_track,phi_shot,phi_track)
            if newDeltaR<DeltaR: 
                index_track = itrack
        return index_track
    #-----------------------------------------------------------
    def getClosestTrackToShot_Variable(self,shot_index,track_index):
        """Return DeltaR, DeltaEta and DeltaPhi between the closest track to the shot and the shot""" 
        pt_shot  = self._tree.tau_pi0Bonn_TauShot_Shot_pt [self._recoIndex][shot_index]
        eta_shot = self._tree.tau_pi0Bonn_TauShot_Shot_eta[self._recoIndex][shot_index]
        phi_shot = self._tree.tau_pi0Bonn_TauShot_Shot_phi[self._recoIndex][shot_index]

        pt_track  = self._tree.tau_track_atTJVA_pt [self._recoIndex][track_index]
        eta_track = self._tree.tau_track_atTJVA_eta[self._recoIndex][track_index]
        phi_track = self._tree.tau_track_atTJVA_phi[self._recoIndex][track_index]

        DeltR   = rawDeltaR(eta_shot,eta_track,phi_shot,phi_track)
        DeltEta = eta_shot-eta_track
        DeltPhi  = DeltaPhi(phi_shot,phi_track)
        Efrac = 0
        if (pt_track*TMath.CosH(eta_track))!=0:
            Efrac = (pt_shot*TMath.CosH(eta_shot))/(pt_track*TMath.CosH(eta_track))
        return DeltR,DeltEta,DeltPhi,Efrac
    #-----------------------------------------------------------
    def getLeadingSubleadingShots_Variable(self):
        """Return DeltaR, DeltaEta and DeltaPhi between the two most energetic shots""" 
        DeltaR   = self.pi0Bonn_TauShot_shots[0].DeltaR  (self.pi0Bonn_TauShot_shots[1])
        DeltaEta = fabs( self.pi0Bonn_TauShot_shots[0].Eta()-self.pi0Bonn_TauShot_shots[1].Eta())
        DeltaPhi = self.pi0Bonn_TauShot_shots[0].DeltaPhi(self.pi0Bonn_TauShot_shots[1])
        return DeltaR,DeltaEta,DeltaPhi

    #-----------------------------------------------------------
    def getInvarMass(self,Entries=-1):
        """Get invariant mass of first 'Entries' entries in shots list"""
        if Entries==-1:
            Entries=self.pi0Bonn_TauShot_nShot
        fullfourvector = TLorentzVector()
        for i in xrange(Entries):
            fullfourvector+=self.pi0Bonn_TauShot_shots[i]
        return fullfourvector.M()

    #-----------------------------------------------------------
    def getEfrac_tot(self,ind):
        """Get fraction of energy in a given cluster compared to the total reconstructed tau energy"""
        return ( self.reco4Vector.E()**(-1) )*self.pi0Bonn_TauShot_shots[ind].E()

    # #-----------------------------------------------------------
    # def getEdensity(self,jetlist,ind):
    #     """Get (1-D) density of energy with respect to eta"""
    #     etamin,etamax=getClusterRange(self._jettree,jetlist['i'][ind])
    #     return jetlist['e'][ind]*(etamax-etamin)**(-1)

    # #-----------------------------------------------------------
    # def getTrackE(self):
    #     """Get (1-D) density of energy with respect to eta"""
    #     # self._tree.
    #     return jetlist['e'][ind]*(etamax-etamin)**(-1)






    # def getTrackVars(self,jetlist,Entries=0):
    #     """Get invariant mass of first 'Entries' entries in jetlist"""
    #     if Entries==0:
    #         Entries=len(jetlist['e'])
    #     etalist=[]
    #     philist=[]
    #     Rlist=[]
    #     trackE=[]
    #     for i in xrange(Entries):
    #         if self.nTracks==0:
    #             etalist.append(0)
    #             philist.append(0)
    #             Rlist.append(0)
    #             trackE.append(0)
    #             continue
    #         Rlist.append( rawDeltaR( jetlist['eta'][i],self.tracks[0].Eta(),jetlist['phi'][i],self.tracks[0].Phi() ) )
    #         philist.append( DeltaPhi( jetlist['phi'][i],self.tracks[0].Phi() ) )
    #         etalist.append( abs( jetlist['eta'][i]-self.tracks[0].Eta() ) )
    #         trackE.append( jetlist['e'][i]*( self.tracks[0].E()**(-1) ) )
    #         for j in self.tracks:
    #     	newR=rawDeltaR( jetlist['eta'][i],j.Eta(),jetlist['phi'][i],j.Phi() )
    #     	if newR<Rlist[i]:
    #     	    Rlist[i]=newR
    #     	    philist[i]=DeltaPhi( jetlist['phi'][i],j.Phi() )
    #     	    etalist[i]=abs( jetlist['eta'][i]-j.Eta() )
    #     	    trackE[i]=( jetlist['e'][i]*( j.E()**(-1) ) )
		
	
    #     return Rlist,etalist,philist,trackE

    # #-----------------------------------------------------------
    # def getGeoCenter(self,jetlist):
    #     """Get geometric center of the strip layer cells"""
    #     N=len(jetlist['e'])
    #     if N==0:
    #         return 0,0
    #     eta_av=sum(jetlist['eta'])*N**(-1)
    #     if max(jetlist['phi'])>3. and min(jetlist['phi'])<-3.:
    #         philist=[phi for phi in jetlist['phi'] if phi > 0]
    #         philist+=[phi+2*pi for phi in jetlist['phi'] if phi < 0]
    #     else:
    #         philist=[phi for phi in jetlist['phi']]
    #     phi_av=sum(philist)*N**(-1)
    #     if phi_av>pi:
    #         phi_av-=2*pi
    #     return eta_av,phi_av

    # #-----------------------------------------------------------
    # def getEnergyCenter(self,jetlist):
    #     """Get energy weighted center of the strip layer cells"""
    #     E=sum(jetlist['e'])
    #     if E==0:
    #         return 0,0
    #     eta_av=sum([e*eta for (e,eta) in zip(jetlist['e'],jetlist['eta'])])*E**(-1)
    #     if max(jetlist['phi'])>3. and min(jetlist['phi'])<-3.:
    #         philist=[phi for phi in jetlist['phi'] if phi > 0]
    #         philist+=[phi+2*pi for phi in jetlist['phi'] if phi < 0]
    #     else:
    #         philist=[phi for phi in jetlist['phi']]
    #     phi_av=sum([e*phi for (e,phi) in zip(jetlist['e'],philist)])*E**(-1)
    #     if phi_av>pi:
    #         phi_av-=2*pi
    #     return eta_av,phi_av

    # #-----------------------------------------------------------
    # def getEfrac_strip(self,jetlist,ind):
    #     """Get fraction of energy in a given cluster compared to the remaining strip layer energy"""
    #     Estrip=sum(jetlist['e'])
    #     return ( Estrip**(-1) )*jetlist['e'][ind]


    # #-----------------------------------------------------------
    # def getDeltaEta_center(self,jetlist,ind):
    #     """Get the eta distance from the event energy weighted center for a given cluster"""
    #     etacenter,phicenter=self.getEnergyCenter(jetlist)
    #     return abs(etacenter - jetlist['eta'][ind])

    # #-----------------------------------------------------------
    # def getDeltaPhi_center(self,jetlist,ind):
    #     """Get the phi distance from the event energy weighted center for a given cluster"""
    #     etacenter,phicenter=self.getEnergyCenter(jetlist)
    #     return min( abs(phicenter - jetlist['phi'][ind]), abs(2*pi+phicenter - jetlist['phi'][ind]), abs(phicenter - jetlist['phi'][ind]-2*pi) )

    # #-----------------------------------------------------------
    # def getDeltaR_center(self,jetlist,ind):
    #     """Get the total distance from the event energy weighted center for a given cluster"""
    #     etacenter,phicenter=self.getEnergyCenter(jetlist)
    #     return ( (etacenter - jetlist['eta'][ind])**2 + min( abs(phicenter - jetlist['phi'][ind]), abs(2*pi+phicenter - jetlist['phi'][ind]), abs(phicenter - jetlist['phi'][ind]-2*pi) )**2 )**0.5

        
    # #-----------------------------------------------------------
    # def getEffEnergyRadius(self,jetlist):
    #     """Get effective energy weighted radius"""
    #     E=0
    #     ER=0
    #     etacenter,phicenter=self.getEnergyCenter(jetlist)
    #     for ie,ieta,iphi in zip(jetlist['e'],jetlist['eta'],jetlist['phi']):
    #         E+=ie
    #         ER+=ie*( (etacenter - ieta)**2 + min( abs(phicenter - iphi),abs(2*pi+phicenter - iphi), abs(phicenter - iphi - 2*pi) )**2 )**0.5
    #     if E>0:
    #         return ER*( E**(-1) )
    #     else:
    #         return 0

    # #-----------------------------------------------------------
    # def getEffGeoRadius(self,jetlist):
    #     """Get effective geometric weigthed radius"""
    #     N=0
    #     R=0
    #     etacenter,phicenter=self.getGeoCenter(jetlist)
    #     for ie,ieta,iphi in zip(jetlist['e'],jetlist['eta'],jetlist['phi']):
    #         if not(ie==0):
    #         	N+=1
    #         	R+=( (etacenter - ieta)**2 + min( abs(phicenter - iphi),abs(2*pi+phicenter - iphi), abs(phicenter - iphi - 2*pi) )**2 )**0.5
    #     if N>0:
    #         return R*( N**(-1) )
    #     else:
    #         return 0

    # #-----------------------------------------------------------
    # def getjetlist(self):
    #     """Get list of jets"""
    #     jetlist={}
    # 	jetlist['i']=[i for (e,i) in sorted(zip(self._jettree.AntiKt4_e,range(self._jettree.AntiKt4_N)), reverse=True)]
    #     jetlist['e']=[ self._jettree.AntiKt4_e[i] for i in jetlist['i']]
    # 	jetlist['eta']=[ self._jettree.AntiKt4_eta[i] for i in jetlist['i']]
    # 	jetlist['phi']=[ self._jettree.AntiKt4_phi[i] for i in jetlist['i']]
    #     jetlist['m']=[ self._jettree.AntiKt4_mass[i] for i in jetlist['i']]
    #     jetlist['pt']=[ self._jettree.AntiKt4_pt[i] for i in jetlist['i']]
    #     jetlist['eta_range']=[getClusterRange(self._jettree,i) for i in jetlist['i']]
    # 	jetlist['phi_range']=[((i-0.05),(i+0.05)) for i in jetlist['phi']]
    # 	jetlist['score']=[]
	
    #     for (i,_) in enumerate(jetlist['phi']):
    #     	if jetlist['phi'][i]>pi:
    #     		jetlist['phi'][i]-=2*pi

    #     #Declare starting number of jets
    #     nJets=len(jetlist['i'])
    
    #     E_strip=sum(jetlist['e'])
    #     tot_score=0
    #     a=1.1
    #     b=0.05
    #     c=1.0
    #     ecut=100
    #     for i in xrange(nJets):
    #         score_i=0
    #         E_i=jetlist['e'][i]
    #         eta_min_i,eta_max_i=jetlist['eta_range'][i]
    #         phi_min_i, phi_max_i=jetlist['phi_range'][i]
    #         if E_i<ecut:
    #             jetlist['score'].append(1000.)
    #             continue
    #         for j in xrange(nJets):
    #             E_j=jetlist['e'][j]
    #             if E_j<ecut:
    #                 continue
    #             eta_min_j,eta_max_j=jetlist['eta_range'][i]
    #             phi_min_j,phi_max_j=jetlist['phi_range'][i]
    #             Delta_eta=max(eta_max_i,eta_max_j) - min(eta_min_i,eta_min_j)
    #             Delta_phi=max( min( abs(phi_max_i - phi_min_j), abs(phi_max_i - phi_min_j-2*pi), abs(phi_max_i - phi_min_j+2*pi) ), min( abs(phi_max_j - phi_min_i), abs(phi_max_j - phi_min_i-2*pi), abs(phi_max_j - phi_min_i+2*pi) ) )
    #             if i==j:
    #                 sum_e=jetlist['e'][i]
    #             else:
    #                 sum_e=jetlist['e'][i]+jetlist['e'][j]
    #             frac_e=E_strip*( sum_e**(-1) )
    #             score_j=(Delta_eta*a)+(Delta_phi*b)+(frac_e*c)
    #             score_i+=score_j
    #             if j>=i:
    #                 tot_score+=score_j
    #         jetlist['score'].append(score_i)
    #     jetlist['totscore']=tot_score

    #     return jetlist
    # # --------------------------------------
    # def getAntiKtShots(self):
    #     """Return the shots found by Matthew's Antikt shot finding alg"""
    #     n = self._jettree.AntiKt4_N
    #     shots = []
    #     for i in range(0,n):
    #         mass = self._jettree.AntiKt4_mass[i]
    #         pt   = self._jettree.AntiKt4_pt  [i]
    #         eta  = self._jettree.AntiKt4_eta [i]
    #         phi  = self._jettree.AntiKt4_phi [i]
    #         if phi>pi: phi = phi-2*pi

    #         vector = TLorentzVector()
    #         vector.SetPtEtaPhiM(pt,eta,phi,mass)
    #         shots.append(vector)
    #     return shots
        

    # #-----------------------------------------------------------
    # def getRecoPi0s(self):
    #     """Get number of reco pi0s"""
    #     Emin=0.006
    #     Epi0=5000
	
    #     #Initialize completion flag
    #     foundAllPi0s=False

    #     #Make Jet list
    # 	jetlist={}
    # 	jetlist['i']=[i for (e,i) in sorted(zip(self._jettree.AntiKt4_e,range(self._jettree.AntiKt4_N)), reverse=True) if self._jettree.AntiKt4_e[i] > Emin*self.reco4Vector.E()]
    #     jetlist['e']=[ self._jettree.AntiKt4_e[i] for i in jetlist['i']]
    # 	jetlist['eta']=[ self._jettree.AntiKt4_eta[i] for i in jetlist['i']]
    # 	jetlist['phi']=[ self._jettree.AntiKt4_phi[i] for i in jetlist['i']]
    # 	#Declare starting number of jets
    # 	nJets=len(jetlist['i'])

    #     #Make pi0 list
    # 	pi0list={}
    #     pi0list['i']=[]
    #     pi0list['e']=[]
    #     pi0list['eta']=[]
    #     pi0list['phi']=[]
    #     pi0list['method']=[]

    #     while not(foundAllPi0s):
    #    		#Check for remaining jets
    #     	if nJets==0:
    #         		foundAllPi0s=True
    #         		break

    #     	#Declare starting number of jets
    #     	nJets=len(jetlist['i'])

    #     	#Find minimum d_ij value
    #     	Discrim=0.00021
    #     	if nJets>=2:
    #         		for i in xrange(1,nJets):
    #             		if discrim( jetlist,0,i,self.reco4Vector.E ) < Discrim:
    #                 			Discrim=discrim( jetlist,0,i,self.reco4Vector.E ) 
    #                 			ind=i 
    #     	#Check for completed pi0 pair and remove other paired gamma from list
    #     	if Discrim<0.00021:
    #     		pi0list['i'].append([jetlist['i'][0],jetlist['i'][ind]])
    #     	        pi0list['e'].append([jetlist['e'][0],jetlist['e'][ind]])
    #     	        pi0list['eta'].append([jetlist['eta'][0],jetlist['eta'][ind]])
    #     	        pi0list['phi'].append([jetlist['phi'][0],jetlist['phi'][ind]])
    #     	        pi0list['method'].append('Pair')
    #     	        jetlist['i']=[jetlist['i'][i] for i in xrange(nJets) if not(i==ind)]
    #     	        jetlist['e']=[jetlist['e'][i] for i in xrange(nJets) if not(i==ind)]
    #     	        jetlist['eta']=[jetlist['eta'][i] for i in xrange(nJets) if not(i==ind)]
    #     	        jetlist['phi']=[jetlist['phi'][i] for i in xrange(nJets) if not(i==ind)]
    #     	        nJets-=1
    #     	#Check if main jet could correspond to tightly clustered pi0 pair
    #     	elif jetlist['e'][0]>Epi0:
    #     		pi0list['i'].append([jetlist['i'][0]])
    #     	        pi0list['e'].append([jetlist['e'][0]])
    #     	        pi0list['eta'].append([jetlist['eta'][0]])
    #     	        pi0list['phi'].append([jetlist['phi'][0]])
    #     	        pi0list['method'].append('Single')

    #     	else:
    #     	    	foundAllPi0s=True
	
    #     	#Remove highest energy jet from list
    #     	jetlist['i']=[jetlist['i'][i] for i in xrange(nJets) if not(i==0)]
    #     	jetlist['e']=[jetlist['e'][i] for i in xrange(nJets) if not(i==0)]
    #     	jetlist['eta']=[jetlist['eta'][i] for i in xrange(nJets) if not(i==0)]
    #     	jetlist['phi']=[jetlist['phi'][i] for i in xrange(nJets) if not(i==0)]
    #     	nJets-=1
    #     for (i,_) in enumerate(pi0list['phi']):
    #     	for (j,_) in enumerate(pi0list['phi'][i]):
    #     		if pi0list['phi'][i][j]>pi:
    #     			pi0list['phi'][i][j]+=pi
    #     return pi0list
