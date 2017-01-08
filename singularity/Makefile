.PHONY: clean singularity

help:
	@echo "clean - remove all build and test artifacts"
	@echo "singularity - Creates singularity image"

clean:
	rm -fr build/

singularity: clean
	@echo 'Creating Singularity image'
	mkdir -p build
	cd ../;python setup.py sdist
	cp -f ../dist/pysegtools*.gz build/.
	vers=`grep "version='" ../setup.py | sed "s/^.*='//" | sed "s/'.*//"` ; \
	echo "Version $$vers"; \
	imgfile=`echo build/segtools-$$vers.img` ; \
	sudo singularity create -s 4096 $$imgfile ; \
	sudo singularity bootstrap $$imgfile centos.def $$vers; \
	echo 'Singularity image created $imgfile'
	ls -l build
