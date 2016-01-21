

nohup ./train --trigger --level hlt --features features --categories training_hlt & 
nohup ./train --trigger --level hlt --features features_pileup_corrected --categories training_hlt &
nohup ./train --features features --categories training & 
nohup ./train --features features_pileup_corrected --categories training &
