clean-pyc:
	find tauperf -name "*.pyc" -exec rm {} \;

clean-tilda:
	find tauperf -name "*~" -exec rm {} \;

clean: clean-pyc clean-tilda
