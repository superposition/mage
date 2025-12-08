.PHONY: build run shell profile stress stress-quick ncu nsys clean help

# Default target
help:
	@echo "Mage GPU Profiler - Docker Commands"
	@echo ""
	@echo "  make build        Build the Docker image"
	@echo "  make shell        Open interactive shell in container"
	@echo "  make profile      Run ncu profiler on script.py"
	@echo "  make stress       Run full GPU stress test"
	@echo "  make stress-quick Run quick GPU stress test"
	@echo "  make ncu SCRIPT=x Run ncu on custom script"
	@echo "  make nsys SCRIPT=x Run nsys on custom script"
	@echo "  make clean        Remove Docker images"
	@echo ""
	@echo "Examples:"
	@echo "  make build && make profile"
	@echo "  make ncu SCRIPT=my_script.py"

# Build the Docker image
build:
	docker compose build

# Open interactive shell
shell:
	docker compose run --rm mage bash

# Run profiler with ncu
profile:
	docker compose run --rm profile

# Run stress tests
stress:
	docker compose run --rm stress

stress-quick:
	docker compose run --rm stress-quick

# Run ncu on custom script (copy script first)
ncu:
ifndef SCRIPT
	$(error SCRIPT is not set. Usage: make ncu SCRIPT=your_script.py)
endif
	docker compose run --rm -v $(PWD)/$(SCRIPT):/app/$(SCRIPT):ro mage \
		mage profile --backend ncu $(SCRIPT)

# Run nsys on custom script
nsys:
ifndef SCRIPT
	$(error SCRIPT is not set. Usage: make nsys SCRIPT=your_script.py)
endif
	docker compose run --rm -v $(PWD)/$(SCRIPT):/app/$(SCRIPT):ro mage \
		mage profile --backend nsys $(SCRIPT)

# Run with all columns
profile-full:
	docker compose run --rm mage \
		mage profile --backend ncu \
		--columns kernel,duration,occupancy,mem_bw,l1_hit,l2_hit,regs,smem \
		script.py

# Clean up
clean:
	docker compose down --rmi local
	docker image prune -f
