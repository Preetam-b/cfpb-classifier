import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def deploy_model():
    role=sagemaker.get_execution_role()
    print(f"Role :{role}")

    hub={
        'HF_MODEL_ID': 'P-r-e-e-t-a-m/cfpb-bert-classifier',
                'HF_TASK':     'text-classification'
    }
    model=HuggingFaceModel(
        env=hub,
        role=role,
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39'
    )

    print("Deploying endpoint")
    predictor=model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )
    print(f"Endpoint deployed : {predictor.endpoint_name}")
    return predictor

def test_endpoint(predictor):
    test_cases=[
         "My credit card company charged me incorrect late fees for 3 months.",
        "My mortgage payment has been incorrectly calculated.",
        "A debt collector keeps calling me after I asked them to stop.",
        "My credit report shows incorrect information affecting my score.",
        "My bank froze my account without any notice."
    ]
    print("\nTest Predictions")
    for test in test_cases:
        result=predictor.predict({"inputs":text})
        label=result[0]['label']
        score=result[0]['score']
        print(f"Text  : {text[:50]}...")
        print(f"Label : {label} ({score:.1%})")

def delete_endpoint(predictor):
    predictor.delete_endpoint()
    print("Endpoint deleted sucessfully")

if __name__=="__main__":
    predictor=deploy_model()
    test_endpoint(predictor)
    delete_endpoint(predictor)