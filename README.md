# Transformers in Java

Transformers-based experimental AI Models written in Java based on [DeepLearning4J](https://deeplearning4j.konduit.ai/) framework. The repository is located [here](https://github.com/partarstu/transformers-in-java).


## Introduction

This project is an experimental work in the field of Artificial Intelligence and Natural Language Processing (NLP). It aims to implement 
and explore the models based on the [Transformer Architecture](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) with 
different modifications aimed to enhance the overall models efficiency. The project is written in Java and utilizes the DeepLearning4J 
framework's [Samediff](https://deeplearning4j.konduit.ai/samediff/tutorials/quickstart) layers as the core of neural networks which 
stand behind each of the models implemented in this project.


## Features

The project includes the following features:

1. Configurable layers which allow to build the transformer block (attention layers and feed-forward layers)
2. Configurable blocks which allow to build the model itself (encoder and decoder)
3. Configurable models themselves. For now only Masked Language (Encoder) Model, Auto-Regressive (Generative) Model and 
   Question-Answering Generative Model (still WIP)
4. Sample classes for models training
5. Sample IntelliJ run configurations for samples and building the Docker Images


## Work in Progress

Please note that this project is still a work in progress. While the core functionality is implemented, there are ongoing developments 
and improvements being made. As a result, the documentation is not yet complete, and some parts of the project may lack detailed explanations.
There are currently also no unit/integration tests implemented due to insufficient resources.


## Usage

To use the project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the latest Java version installed on your machine.
3. Open the project in your IDE.
4. Explore the existing code and models to understand their structure and functionality.
5. Modify and extend the project to suit your specific needs.
6. Provide the required environment variables before running any sample based on the README documentation.

[This Readme file](train//README.md) is an entry point which provides a lot of information on how to run the model training and/or inference.


## Contributing

Contributions to the project are welcome. If you find any issues, have suggestions for improvements, or would like to add new features, 
please feel free to submit a pull request. However, due to the project being a work in progress, please ensure that your contributions 
align with the project's direction and goals. In order to avoid formatting issues, please use the cody style formatter config in 
[code_style](code_style) folder.


## Notes

### Documentation 
As mentioned earlier, the documentation for this project is still under development. While some information and usage instructions are provided, please be aware that not all aspects of the project may be thoroughly documented at this stage.

### Comments 
The project currently does not have many comments within the code. However, efforts are being made to improve code readability and add comments to enhance understanding.


## License

The project is released under the Apache 2.0 License. Feel free to modify and distribute the code within the terms of the license.


## Acknowledgments

This project is inspired by the original "Attention Is All You Need" [paper](https://arxiv.org/abs/1706.03762) and is built upon the DeepLearning4J framework. 

I'd like to express my gratitude to all the authors of the original "Attention Is All You Need" [paper](https://arxiv.org/abs/1706.03762)
as well as to the creators and contributors of DeepLearning4J framework.


## Contact

For any inquiries or additional information regarding the project, please contact me via e-mail. 

In case you notice any bugs or you have any enhancement suggestions - please create an issue. I'm checking those at a regular basis and will try to answer as soon as possible.
