.PHONY: all
all: build install


.PHONY: build
build:
	@python setup.py build_ext


.PHONY: install
install:
	@pip install -e .


JUNK = "*.egg-info" "__pycache__" "*.c" "*.so" "build"

.PHONY: clean
clean:
	$(foreach fd, $(JUNK), rm -rf `find . -name $(fd)`;)
