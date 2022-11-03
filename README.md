# dreambooth

Attempt to make Dreambooth work on less than 16Go AWS G4 & P3 instances (Nvidia T4 and P100 GPUs)

Update: It works on a Sagemaker Notebook, using the script that I provided.

TODO:

- Make it work on a Notebook in sagemaker ✅
- Container for the training
- Try to make it compatible with Sagemaker for training and inference
-  Cloudformation template to deploy the MLOps pipeline (Step Function workflow)

Workflow:

- Takes the code from codecommit and builds the docker image
- Takes the code from codecommit and builds the StepFunction 
-  If a new subdirectory with a configuration file is created in the bucket, then we trigger the pipeline for training (New subject to train a model on)
- Training job is triggered with the right parameters
- Model is saved and sample images are generated 
- Review the generated images and decide to publish the model or not
- Publish the model as an endpoint or launch 1 machine and a queue to dynamically generate images no matter the model (Load the model on the fly)

Extra: 

- Build a web app with amplify to do all of this in an easy way or adapt the SD web UI for this project

