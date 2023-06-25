/*
 * Copyright 2023 Taras Paruta
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package org.tarik.train;

import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.question_answering.GenerativeQuestionAnsweringModel;
import org.tarik.core.network.models.transformer.question_answering.sequence.IQaTokenSequence;
import org.tarik.core.vocab.WordPieceVocab;
import org.tarik.train.qa.sequence.MultipleContextsQaTokenSequence;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static java.lang.Integer.parseInt;
import static java.lang.System.exit;
import static java.lang.System.getenv;
import static java.nio.file.Files.lines;
import static java.util.stream.Collectors.toCollection;
import static org.tarik.core.vocab.WordPieceVocab.loadVocab;
import static org.tarik.utils.ResourceLoadingUtil.getResourceFileStream;

public class TrainGenerativeQa extends CommonTrainer {
    private static final Logger LOG = LoggerFactory.getLogger(TrainGenerativeQa.class);
    private static final Random random = new Random();
    private static final Path ROOT_PATH = Paths.get("D:/ai_tuning/save");
    private static final Path MODEL_PATH = ROOT_PATH.resolve("generative_qa_model");
    private static final Path BACKUP_PATH = ROOT_PATH.resolve("generative_qa_model_backup");
    private static final Path QA_TEST_DATASET_PATH = ROOT_PATH.resolve("test_multicontext_qa_tokens_list.json");
    private static final Path QA_ALL_DATASET_PATH = ROOT_PATH.resolve("all_multicontext_qa_tokens_list.json");
    private static final Gson gson = new Gson();
    private static final int TARGET_PASSAGES_AMOUNT = 10;
    private static final int DECODER_BATCH_SIZE = 10;
    private static final int ENCODER_BATCH_SIZE = TARGET_PASSAGES_AMOUNT * DECODER_BATCH_SIZE;
    private static final double LEARNING_RATE = 1e-5;
    private static final int EPOCHS_AMOUNT = 5000;
    private static final int TOTAL_TRAIN_QA_ITEMS = 100000;
    private static final int STEPS_PER_EPOCH = TOTAL_TRAIN_QA_ITEMS / DECODER_BATCH_SIZE;
    private static final int ALL_STEPS_AMOUNT = STEPS_PER_EPOCH * EPOCHS_AMOUNT;
    private static final int LOG_FREQ = parseInt(getenv().getOrDefault("log_freq", "10"));
    private static final int TESTING_FREQ = parseInt(getenv().getOrDefault("test_freq", "20"));
    private static final int ENCODER_LAYERS_AMOUNT = parseInt(getenv().getOrDefault("layers", "4"));
    private static final int SAVE_FREQ = parseInt(getenv().getOrDefault("save_freq", "50"));

    private static GenerativeQuestionAnsweringModel getModel(WordPieceVocab vocab) {
        return new GenerativeQuestionAnsweringModel.Builder(vocab)
                .withLoggingFrequency(LOG_FREQ)
                .withSaveFrequencyInSteps(SAVE_FREQ)
                .withModelTestFrequency(TESTING_FREQ)
                .withEpochsAmount(EPOCHS_AMOUNT)
                .withEncoderBatchSize(ENCODER_BATCH_SIZE)
                .withLearningRate(LEARNING_RATE)
                .withEncoderSequenceLength(256)
                .withSequenceLength(20)
                .withAmountOfPassagesPerAnswer(TARGET_PASSAGES_AMOUNT)
                .withAttentionHeadsAmount(8)
                .withEncoderLayersAmount(ENCODER_LAYERS_AMOUNT)
                .withOwnLayersAmount(ENCODER_LAYERS_AMOUNT)
                .withLabelSmoothing(0.1)
                .build();
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, CloneNotSupportedException {
        prepareEnvironment();
        startMemoryProfiling(60);

        try {
            addModelSaveSafeShutdownHook();
            WordPieceVocab vocab = loadVocab(getResourceFileStream("vocab.txt", GenerativeQuestionAnsweringModel.class));
            var qaTransformer = getModel(vocab);
            var testData = loadQaTokenSequences();
            var trainData = loadQaTokenSequencesStream(QA_ALL_DATASET_PATH)
                    .filter(qaTokenSequence -> qaTokenSequence.getAnswerTokens().length < 20)
                    .filter(qaTokenSequence -> !testData.contains(qaTokenSequence))
                    .limit(1000)
                    .collect(toCollection(ArrayList::new));            LOG.info("Loaded {} sequences for training", trainData.size());

            qaTransformer.train(trainData,
                    model->saveModel(model, MODEL_PATH, BACKUP_PATH),
                    ALL_STEPS_AMOUNT,
                    model -> model.test(
                            random.ints(0, testData.size()).distinct().limit(4).mapToObj(testData::get).toList()));

            LOG.info("Completed the whole training");
            System.exit(0);
        } catch (Throwable t) {
            LOG.error("ERROR intercepted", t);
            exit(1);
        }
    }

    private static Stream<IQaTokenSequence> loadQaTokenSequencesStream(Path file) throws IOException {
        return lines(file)
                .map(jsonString -> gson.fromJson(jsonString, MultipleContextsQaTokenSequence.class));
    }

    private static List<IQaTokenSequence> loadQaTokenSequences() throws IOException {
        return loadQaTokenSequencesStream(TrainGenerativeQa.QA_TEST_DATASET_PATH).toList();
    }
}