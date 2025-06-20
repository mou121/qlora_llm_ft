from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import os

role = os.environ["SAGEMAKER_ROLE"]
bucket = os.environ.get("S3_BUCKET", "your-s3-bucket")

# Path to model.tar.gz created after merging
model_path = f"s3://{bucket}/qlora-mistral-model/model.tar.gz"

huggingface_model = HuggingFaceModel(
    model_data=model_path,
    role=role,
    transformers_version="4.40.0",
    pytorch_version="2.1.1",
    py_version="py310",
    env={"HF_TASK": "text-generation"}
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)

response = predictor.predict({"inputs": "Explain LoRA."})
print(response)
