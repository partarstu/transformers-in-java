## Train module

This module contains the classes which represent the model trainers. Each class defines all parameters which are required to train and 
test the model (during training).

### Model Training

Each model has its own trainer class example:
1. [TrainMlm.java](src/main/java/org/tarik/train/TrainMlm.java) for Masked Language (Encoder) Model
2. [TrainGenerator.java](src/main/java/org/tarik/train/TrainGenerator.java) for Auto-Regressive (Generative) Model
3. [TrainGenerativeQa.java](src/main/java/org/tarik/train/TrainGenerativeQa.java) for Question-Answering Generative Model (still WIP)

Those model trainers require different environment variables to be present, but almost all of them have fall-back default values. The 
variables themselves are quite similar from model to model. [TrainMlm.java](src/main/java/org/tarik/train/TrainMlm.java) is the only one 
which has all variables commented. 

IntelliJ run configurations for each of those classes could be found in the [.run](..//.run) folder. The environment variables which 
need to be provided explicitly:
1. **OMP_NUM_THREADS** - defines the amount of CPU cores which the current system has. This value is crucial to the speed of the 
   training/inference.
2. **root_path** - defines the local folder which will be used in order to load and store the saved files. This folder should contain e.g. 
   the test dataset file in order to test the model's accuracy during training (example of such files are located in
   [TrainMlm.java](src/main/resources/test_data) folder).

In order to successfully run the training of Masked Language (Encoder) and Auto-Regressive (Generative) models, you need to provide 
your own `IDataProvider` instance and set it using the `transformer.setDataProvider()` method. The current implementation uses 
`WikiArticlesContentProvider`, which relies on a corresponding MongoDB running with a Wikipedia dump. That was the primary data source 
for training, but because setting it up is quite tedious, the fastest option is to implement your own `IDataProvider` as a lambda or 
anonymous class that fetches the data you need. Please refer to the method `org.tarik.train.TrainMlm#getWikiArticlesContentProvider` for 
guidance on fetching the data in chunks. The implementation of `getPassages(Function<List<String>, Boolean> isLimitReachedFunction)`
method is very important. This one is called depending on how much data you provide to the model. The latter expects you to provide at 
least `BATCH_SIZE` (default=128) sequences (passages) of tokens for one iteration. There is also `MIN_SEQUENCE_UTILIZATION` (default=50%)
variable which tells how many tokens in % from the sequence length (hardcoded as 256 for now) each sequence from the provider should 
contain at least, so that it could be accepted by the model and added to the `batchedTokenSentences` variable. If your provider gives 
back per one call less than `BATCH_SIZE` sequences, the model will call it so many times, till it gets the whole batch. In a similar way, 
if the sequences which come from provider contain less than 50% of tokens (the rest will always be masked), the model skips (ignores) them 
and calls again `getPassages(...)` as many times as needed in order to fill the batch. This could be the source of eternal loop, if the 
provider always gives back some data and gets never exhausted (e.g. is a simple mock).

### Logging

If you want to see more detailed logs - you could change the LOG_LEVEL environment variable value to DEBUG (it's INFO by default). Also, 
you can turn on the javacpp debug (if anything's wrong with C++ - related part) using the following code:
`System.setProperty("org.bytedeco.javacpp.logger.debug", "true")`. There's also SameDiff logging which allows to see details of each 
running operation. You can turn it on using this line : `sd.enableDebugMode()`;

### Docker Setup

In order to create a docker image for Linux, the following IntelliJ run configurations could be used:
1. [create_image_mlm_linux_avx2.run.xml](..//.run//create_image_mlm_linux_avx2.run.xml) - creates an image using [openjdk slim image](https://hub.docker.com/_/openjdk)
   based on the [dockerfile_mlm](..//docker//dockerfile_mlm) docker file using **AVX2** extensions. 
2. [create_image_mlm_linux_avx_512.run.xml](..//.run//create_image_mlm_linux_avx_512.run.xml) - creates an image using [openjdk slim image]
   (https://hub.docker.com/_/openjdk) based on the [dockerfile_mlm](..//docker//dockerfile_mlm) docker file using **AVX512** extensions.

Docker must be installed on the PC, where those run configurations are executed. Each of those run configurations creates the image, 
installs it into the local registry and exports it as a TAR archive locally. This archive can be later copied anywhere and used in 
order to start the container.

All Docker-related files are located in the [docker](..//docker) folder:
1. [dockerfile_mlm](..//docker//dockerfile_mlm) docker file example could be used for starting the model training using openjdk slim image.
2. [run_mlm_model_training.sh](..//docker//run_mlm_model_training.sh) shell script is an example of starting the docker container on 
   Linux from generated by any linux run configuration docker image archive (tested on Debian OS). This script provides insights into 
   how the Docker container is started and how environment variables are passed to it.
3. [start_training.service](..//docker//start_training.service) is an example of a service file for Linux which starts the model 
   training after system restart automatically (is useful for cloud VMs which are frequently restarted, like spot VMs in Google Cloud)

To start the Docker container, you can use the `docker run` command.


### Debugging

There's a very valuable listener class in the model class itself - CustomListener. This one has different methods which allow you to see 
what happens during the execution. For example, the method 
`public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs)` allows you to see 
the results and params of each operation after it's been executed. It will allow you to understand if the model is training at all or if 
it's stuck somewhere. Because it's quite hard to debug SameDiff directly, this listener is a good utility to see what happens during 
training. It also has methods which allow to see which weights are updated, as well as to see what happens before each operation is executed.

You can use that listener almost always, even if I want to check that the intermediate state of your operations is as expected.