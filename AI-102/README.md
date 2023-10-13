## Plan and Manage an Azure AI Solution (15%)
### Define artificial intelligence

| Capability                                                                                                                                                                                                                                             | Description                                                                                                                                                                                                                                                                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Visual perception                                                                                                                                                                                                                                      | The ability to use computer vision capabilities to accept, interpret, and process input from images, video streams, and live cameras.                                                                                                                                                                                                                                                    |
| Text analysis                                                                                                                                                                                                                                          | The ability to use natural language processing (NLP) to not only "read", but also extract semantic meaning from text-based data.                                                                                                                                                                                                                                                         |
| Speech                                                                                                                                                                                                                                                 | The ability to recognize speech as input and synthesize spoken output. The combination of speech capabilities together with the ability to apply NLP analysis of text enables a form of human-compute interaction that's become known as conversational AI, in which users can interact with AI agents (usually referred to as bots) in much the same way they would with another human. |
| Decision making |  The ability to use past experience and learned correlations to assess situations and take appropriate actions. For example, recognizing anomalies in sensor readings and taking automated action to prevent failure or system damage.                                                                                                                                                   |
### Understand AI-related terms
AI/ML/DS
### Understand considerations for AI Engineers
Model training and inferencing/Probability and confidence scores/Responsible AI and ethics
### Understand considerations for responsible AI
Fairness/Reliability and safety/Privacy and security/Inclusiveness/Inclusiveness/Accountability

### Understand capabilities of Azure Machine Learning

|Feature	|Capability|
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Automated machine learning|	This feature enables non-experts to quickly create an effective machine learning model from data.|
|Azure Machine Learning designer|	A graphical interface enabling no-code development of machine learning solutions.|
|Data and compute management|	Cloud-based data storage and compute resources that professional data scientists can use to run data experiment code at scale.|
|Pipelines	|Data scientists, software engineers, and IT operations professionals can define pipelines to orchestrate model training, deployment, and management tasks.|

***Data scientists*** can use Azure Machine Learning throughout the entire machine learning lifecycle to:

* Ingest and prepare data.
* Run experiments to explore data and train predictive models.
* Deploy and manage trained models as web services.

***Software engineers*** may interact with Azure Machine Learning in the following ways:

* Using Automated Machine Learning or Azure Machine Learning designer to train machine learning models and deploy them as REST services that can be integrated into AI-enabled applications.
* Collaborating with data scientists to deploy models based on common frameworks such as Scikit-Learn, PyTorch, and TensorFlow as web services, and consume them in applications.
* Using Azure Machine Learning SDKs or command-line interface (CLI) scripts to orchestrate DevOps processes that manage versioning, deployment, and testing of machine learning models as part of an overall application delivery solution.

### Understand capabilities of Azure AI Services
Natural language processing/	Knowledge mining and document intelligence/	Computer vision	Decision support/	Generative AI
### Understand capabilities of the Azure Bot Service
AI engineers can develop Bots by writing code, using the classes available in the **Bot Framework SDK**. Alternatively, you can use the **Bot Framework Composer** to develop complex bots using a visual design interface.

### Understand capabilities of Azure Cognitive Search
Azure Cognitive Search is an Applied AI Service that enables you to ingest and index data from various sources, and search the index to find, filter, and sort information extracted from the source data.

## Create and consume Azure AI Services
Azure AI Services include a wide range of AI capabilities that you can use in your applications. To use any of the AI Services, you need to create appropriate resources in an Azure subscription to define an endpoint where the service can be consumed, provide access keys for authenticated access, and to manage billing for your application's usage of the service.

### Provision Azure AI Services resources in an Azure subscription.
create appropriate resources in an Azure subscription to define an endpoint where the service can be consumed, provide access keys for authenticated access, and to manage billing for your application's usage of the service.:
* Multi-service resource: manage a single set of access credentials to consume multiple services at a single endpoint, and with a single point of billing for usage of all services.
* Single-service resource:use separate endpoints for each service (for example to provision them in different geographical regions) and to manage access credentials for each service independently. It also enables you to manage billing separately for each service.

Training and inferencing models use separate dedicated service-specific resource and a generic Azure AI Services resource.
### Identify endpoints, keys, and locations required to consume an Azure AI Services resource.
When you provision an Azure AI service resource in your Azure subscription, you are defining an endpoint through which the service can be consumed by an application.

To consume the service through the endpoint, applications require the following information:

* The endpoint URI. This is the HTTP address at which the REST interface for the service can be accessed. Most AI Services software development kits (SDKs) use the endpoint URI to initiate a connection to the endpoint.
* A subscription key. Access to the endpoint is restricted based on a subscription key. Client applications must provide a valid key to consume the service. When you provision an AI Services resource, two keys are created - applications can use either key. You can also regenerate the keys as required to control access to your resource. 
* The resource location. When you provision a resource in Azure, you generally assign it to a location, which determines the Azure data center in which the resource is defined. While most SDKs use the endpoint URI to connect to the service, some require the location.
### Use a REST API to consume an Azure AI service.
Azure AI Services provide REST application programming interfaces (APIs) that client applications can use to consume services. In most cases, service functions can be called by submitting data in JSON format over an HTTP request, which may be a POST, PUT, or GET request depending on the specific function being called. The results of the function are returned to the client as an HTTP response, often with JSON contents that encapsulate the output data from the function.


![restapi.png](img%2Frestapi.png)
### Use an SDK to consume an Azure AI service.
Software development kits (SDKs) for common programming languages abstract the REST interfaces for most Azure AI Services. 
![sdk.png](img%2Fsdk.png)

## Secure Azure AI Services
### Consider authentication

1. **Subscription keys** are to get access to **Azure AI services resources**.
Management of access to these **keys** is a primary consideration for security. 
2. Regenerate keys:
* via Azure portal: using the visual interface
* Azure command-line interface (CLI) command: using the az cognitiveservices account keys regenerate 
3. Consideration of avoiding service interruption when regenerate keys:

* Configure all production applications to use key 2.
* Regenerate key 1
* Switch all production applications to use the newly regenerated key 1.
* Regenerate key 2.

4. Protect keys:
5. 
5. 
## Create Computer Vision Solutions (20%)

### Create computer vision solutions using Azure Computer Vision.
### Implement image analysis solutions.

## Implement Knowledge Mining Solutions (20%)

### Create a QnA Maker knowledge base.
### Create a language model using Azure Machine Learning.
### Create a custom document search solution using Azure Cognitive Search.

## Monitor and Optimize an AI Solution (15%)

### Monitor and evaluate the performance of an AI solution.
### Optimize and improve the performance of models and solutions.
## Implement a Recommender Solution (15%)

### Create a recommendation model.
### Deploy a recommendation model as an API.