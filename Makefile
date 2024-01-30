WORKDIRNAME = $(shell sh -c "pwd | tr '/' '\\n' | tail -n 1")
EXAMPLES = /opt/uut/examples

DOCKERRUN = docker run -ti ${WORKDIRNAME}_build_test 

PROGRESS = --progress plain

.PHONY: pip_test test

test_build:
	docker-compose build ${PROGRESS} build_test

test_build_no_cache:
	docker-compose build ${PROGRESS} --no-cache build_test


container_shell: test_build
	docker run -ti  ${WORKDIRNAME}_build_test bash

integration_tests: test_build
	${DOCKERRUN} python3 ${EXAMPLES}/demo_one_Fiber.py
	${DOCKERRUN} python3 ${EXAMPLES}/model_single_Muscle.py
	${DOCKERRUN} python3 ${EXAMPLES}/model_test_MUAP_center_tracking.py 
	#${DOCKERRUN} python3 ${EXAMPLES}/motor_unit_pool.py 

unit_test: test_build
	${DOCKERRUN} sh -c \
		"cd tests; python3 -m pytest MUAP_firing_frequenzy_test.py"

	${DOCKERRUN} cargo test

test: unit_test integration_tests
	echo test OK
