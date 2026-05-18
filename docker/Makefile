UBUNTU_VERSION=16.04
DOCKER_IMAGE=ubuntu:$(UBUNTU_VERSION)-sniper-$(USER)
DOCKER_FILE=Dockerfile-ubuntu-$(UBUNTU_VERSION)

DOCKER_EXPERIMENT_IMAGE=ubuntu:$(UBUNTU_VERSION)-sniper-experiments-$(USER)

# Space-separated list of run.py function names — one container is launched per entry.
EXPERIMENTS ?= test_sota

# Absolute path to the results directory on the host, mounted into every container.
RESULTS_DIR ?= $(abspath ../results)

all: $(DOCKER_FILE)
	docker build -f $(DOCKER_FILE) -t $(DOCKER_IMAGE) ../
	
run:
	docker run --privileged --rm -it -v "${HOME}:${HOME}" --user $(shell id -u):$(shell id -g) -w "${PWD}" $(DOCKER_IMAGE)

# Build the self-contained experiment image (bakes in all project files and compiles).
# This image does not require mounting the user's home directory.
build-experiments: $(DOCKER_FILE)
	docker build -f $(DOCKER_FILE) -t $(DOCKER_IMAGE) ../
	docker build \
		--build-arg BASE_IMAGE=$(DOCKER_IMAGE) \
		-f Dockerfile-experiments \
		-t $(DOCKER_EXPERIMENT_IMAGE) \
		../

.PHONY: all
