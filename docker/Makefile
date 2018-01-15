# Docker organization to pull the images from
ORG = stanford

# Name of image
IMAGE = gibson

# Docker TAG
TAG = 0.1

build:
	docker build \
		-t $(ORG)/$(IMAGE):$(TAG) \
		--build-arg IMAGE=$(ORG)/$(IMAGE):$(TAG) \
		--build-arg VCS_REF=`git rev-parse --short HEAD` \
		--build-arg VCS_URL=`git config --get remote.origin.url` \
		--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
		.

example:
	docker build \
		-t $(ORG)/$(IMAGE)-example:$(TAG) example/

push:
	docker push $(ORG)/$(IMAGE):$(TAG)

.PHONY: build example push