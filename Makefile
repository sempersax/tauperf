clean-pyc:
	find tauperf grid batch -name "*.pyc" -exec rm {} \;

clean-tilda:
	find tauperf grid batch -name "*~" -exec rm {} \;

clean: clean-pyc clean-tilda

clean-training:
	find log -name "*.e*" -exec rm {} \;
	find log -name "*.o*" -exec rm {} \;
	find weights -name "*.root" -exec rm {} \;
	find weights -name "*.xml" -exec rm {} \;
	find weights -name "*.C" -exec rm {} \;

train: clean-training
	./bdt-grid-scan