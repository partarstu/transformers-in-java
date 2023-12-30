#!/bin/bash
# Copyright 2023 Taras Paruta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

DATA_DIR="/data/model/mlm"
IMAGE_FILE_NAME="transformer_mlm.tar"
IMAGE_LAST_TIMESTAMP_FILE_NAME="last_image_timestamp.txt"
IMAGE_NAME="transformer_mlm:1.0-SNAPSHOT"
JAR_NAME="train-1.0-SNAPSHOT-shaded.jar"
RUN_COMMAND="java -XX:+UseZGC -Xmx3G --enable-preview -jar ${JAR_NAME}"

cd ${DATA_DIR}

if [ ! -e $IMAGE_LAST_TIMESTAMP_FILE_NAME ]
then
    touch $IMAGE_LAST_TIMESTAMP_FILE_NAME
fi

echo 'Removing all containers ..'
docker rm -f $(docker ps -aq)
echo '----------------------------------------------------'
echo

currentTimestamp=`stat -c %Y ${IMAGE_FILE_NAME}`
previousTimeStamp=$(<${IMAGE_LAST_TIMESTAMP_FILE_NAME})
if [ -z $previousTimeStamp ] || [ "$currentTimestamp" -gt "$previousTimeStamp" ];
then
 echo "There's a newer image TAR file. Loading it"
 echo 'Removing all images ...'
 docker rmi $(docker images -q)
 echo
 echo "Loading the local model docker image ${DATA_DIR}/${IMAGE_FILE_NAME}"
 docker load -i ${DATA_DIR}/${IMAGE_FILE_NAME}
 echo $currentTimestamp > ${IMAGE_LAST_TIMESTAMP_FILE_NAME}
 echo
 echo "Updated the actual image timestamp"
 echo '----------------------------------------------------'
 echo
fi

echo 'Starting the loaded image'
CONTAINER_ID=$(docker run --name mlm_model -d --net=host --restart on-failure --stop-timeout=29 --mount type=bind,source=${DATA_DIR},target=${DATA_DIR} \
-e accuracy=60 \
-e batch_size=196 \
-e cache_percentage=20 \
-e dimensions=768 \
-e epochs=2 \
-e layers=4 \
-e learning_rate=0.0001 \
-e load_model=true \
-e load_updater=true \
-e log_freq=100 \
-e LOG_LEVEL=info \
-e max_jvm_memory=28G \
-e max_memory_log_freq_minutes=300 \
-e min_sequence_utilization=60 \
-e OMP_NUM_THREADS=8 \
-e only_article_summaries=false \
-e pos_attn_active=false \
-e pos_attn_grad_scaling=10 \
-e pos_attn_hyperb_value_scaling=1.5 \
-e prediction_masking_percentage=80 \
-e root_path=${DATA_DIR} \
-e save_freq=15 \
-e test_data_file=test_data.txt \
-e test_freq=200 \
-e use_caching=false \
-e use_cloud_logging=true \
${IMAGE_NAME} ${RUN_COMMAND})

echo '----------------------------------------------------'
echo
echo "Starting log forwarding onto console for container ${CONTAINER_ID}"
docker logs -f ${CONTAINER_ID} &
echo 'ALL DONE'