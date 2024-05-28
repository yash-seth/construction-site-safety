# Efficient Framework for Construction Workers Safety using AI Techniques
Project built as part of the final year Capstone Project.
     
## Modules
- Deep Learning Model for Helmet Detection
     - TTACT formula - image - apply image transformations to improve the performance of object detection model in possible conditions such as low lighting, poor input device etc
     - Overlap module for detection of correct wearing of helmet - video feed - affine invariant approach for helmet detection, which does not require manual annotation work to be done to prepare dataset.
- LLM based RAG pipeline - query system to interact with worker logs using Natural Language queries

## Architecture Diagram
![image](https://github.com/yash-seth/construction-site-safety/assets/71393551/eb61ea2e-cc4f-44f0-9134-c7a8d5d9534a)

## Module-wise Working
### TTACT
![image](https://github.com/yash-seth/construction-site-safety/assets/71393551/4199de7c-9782-475e-a755-7100a86cdb16)

### Overlap Module
#### Diagramatic Representation
![image](https://github.com/yash-seth/construction-site-safety/assets/71393551/cf64273a-3bc7-4653-9761-fdba3c8d6750)

#### Working
![image](https://github.com/yash-seth/construction-site-safety/assets/71393551/da9b341f-cc07-4263-ac53-23d043f1bb31)

### RAG Module
![image](https://github.com/yash-seth/construction-site-safety/assets/71393551/884d143a-7944-4e23-a994-d02d1ed66cdb)


## Technologies Used
- YOLOv8 - Object Detection Model
- PaLM 2 LLM - Large Language Model
- LangChain - framework to interact with LLMs
- FAISS - Facebook AI Similarity Search
- Instructor Text Embedding model
- Python
- Jupyter Notebooks
- React
- Express
