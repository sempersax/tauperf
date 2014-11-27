clean-pyc:
	find tauperf grid batch -name "*.pyc" -exec rm {} \;

clean-tilda:
	find tauperf grid batch -name "*~" -exec rm {} \;

clean: clean-pyc clean-tilda
