PYTHON = python
KNOWLEDGE_DISTILLATION = knowledge_distillation.py

DATA_DIR = ./data/distillation_data
STUDENT_MODEL_TYPE = distilbert
OUTPUT_DIR = ./models/distilled
COREML_OUTPUT_DIR = ./models/coreml
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 2.5e-6
TEMPERATURE = 2.0
ALPHA = 0.7
PATIENCE = 5
FOCAL_GAMMA = 2.0

all: distill

distill:
	$(PYTHON) $(KNOWLEDGE_DISTILLATION) \
		--data_dir "$(DATA_DIR)" \
		--student_model_type "$(STUDENT_MODEL_TYPE)" \
		--output_dir "$(OUTPUT_DIR)" \
		--coreml_output_dir "$(COREML_OUTPUT_DIR)" \
		--batch_size $(BATCH_SIZE) \
		--num_epochs $(NUM_EPOCHS) \
		--learning_rate $(LEARNING_RATE) \
		--temperature $(TEMPERATURE) \
		--alpha $(ALPHA) \
		--patience $(PATIENCE) \
		--focal_gamma $(FOCAL_GAMMA) \
		--use_class_weights \
		--use_focal_loss 

clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(COREML_OUTPUT_DIR)

.PHONY: all distill clean
