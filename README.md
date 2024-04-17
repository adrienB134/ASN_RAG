# RAG_ASN
## 🚧 Building a full RAG application 🚧

The goal is to build a full Retrieval Augmented Generation chatbot for french data from the nuclear safety agency to help researchers and nuclear practitioners. The application will be complete with a feature ingestion pipeline, a training pipeline for finetuning models to better align with the domain, an inference pipeline and an evaluation pipeline.

## Overview of the project
<br> ![image](https://github.com/adrienB134/ASN_RAG/assets/102990337/96757f50-da7f-4b71-a028-6adb30149934)

## Evaluation pipeline

This is the most critical part of the project. It will serve for both model and achitecture evaluation, based on RAGAS evaluation metrics for RAG apps. 
<br> Testset as been generated using RAGAS.
<br><br> **TODO**:
- Write an evaluation script
- Instrument that script with Arize.ai to log the traces and metrics

## Feature ingestion pipeline

For now only the ingestion part is done. Historical data was scraped in a previous project so i'm using that at the moment. 
<br><br> **TODO**:
* Data collection + nosql DB storage
* RabbitMQ queue
* connect the queue and the ingestion in a streaming fashion with bytewax
  

## Training pipeline

Training API is done
<br><br> **TODO**:
* Automate beam deployement
* Create a DSPy program for synthetic dataset generation. Something similar to what I did when re-implementing the [UDAPDR](https://arxiv.org/abs/2303.00807) paper in [RAGatouille](https://github.com/adrienB134/RAGatouille/blob/main/ragatouille/RAGInDomainTrainer.py).


## Inference pipeline
For now a simple multi-query retriever + reranking is implemented for RAG. The chatbot part is done using FastAPI and a basic HTML + JS frontend.<br>
For now gpt-3.5-turbo is used as a llm as it is fast and cheap, plus I don't have a good finetuned model yet.
<br><br> **TODO**:
* the frontend needs a facelift
* Switch to a fine-tuned model when possible

