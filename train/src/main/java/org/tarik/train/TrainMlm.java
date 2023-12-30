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

import com.google.common.collect.ImmutableSet;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.mlm.MlmTransformerSdModel;
import org.tarik.core.network.models.transformer.mlm.MlmTransformerSdModel.Builder;
import org.tarik.core.network.models.transformer.mlm.MlmTransformerSdModel.EncoderType;
import org.tarik.core.vocab.PosTagsVocab;
import org.tarik.core.vocab.WordPieceVocab;
import org.tarik.train.db.model.wiki.WikiTextArticle;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

import static dev.morphia.query.experimental.filters.Filters.gt;
import static java.lang.Boolean.parseBoolean;
import static java.lang.Float.parseFloat;
import static java.lang.Integer.parseInt;
import static java.lang.Long.parseLong;
import static java.lang.String.format;
import static java.lang.System.getenv;
import static java.nio.file.Files.*;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.util.stream.Collectors.toList;
import static org.tarik.core.vocab.WordPieceVocab.loadVocab;
import static org.tarik.train.db.model.wiki.WikiDatastore.fetchArticlesFromDb;
import static org.tarik.utils.CommonUtils.deserializeMapFromJsonWithJackson;
import static org.tarik.utils.ResourceLoadingUtil.getResourceFileLines;
import static org.tarik.utils.ResourceLoadingUtil.getResourceFileStream;

/**
 * A sample of the class which trains the {@link MlmTransformerSdModel}. The following environment variables have to be provided in order
 * to configure the model:
 * <ul>
 * <li><b>LOG_LEVEL</b> <i>(String)</i> - logging level</li>
 * <li><b>OMP_NUM_THREADS</b> <i>(int)</i> - amount of processor threads which will be used for model's training</li>
 * <li><b>root_path</b> <i>(String)</i> - the path to the directory which contains the data required for the training</li>
 * <li><b>test_data_file</b> <i>(String)</i> - the name of the file which contains the test data</li>
 * <li><b>batch_size</b> <i>(int)</i> - model's batch size</li>
 * <li><b>log_freq</b> <i>(int)</i> - the frequency (in iterations) of logging the model's accuracy and other debug-related data</li>
 * <li><b>test_freq</b> <i>(int)</i> - the frequency (in iterations) of testing the model during training using the provided test dataset</li>
 * <li><b>save_freq</b> <i>(int)</i> - the frequency (in iterations) of saving the model locally</li>
 * <li><b>min_sequence_utilization</b> <i>(int)</i> - minimum percentage of the sequence utilization. By default
 * {@link WikiArticlesContentProvider} feeds the model with tokens which form the full sentences. It means that quite often the whole
 * sequence can't be filled with tokens and the rest will be padded. This percentage tells the model to skip any passage which doesn't
 * have at least that percentage of tokens</li>
 * <li><b>epochs</b> <i>(int)</i> - the amount of mini-epochs to iterate through using the same dataset (as usually it's 1)</li>
 * <li><b>max_memory_log_freq_minutes</b> <i>(int)</i> - the frequency (in minutes) of logging the current memory usage by the model.
 * Could be useful for detecting memory leaks</li>
 * <li><b>dimensions</b> <i>(int)</i> - the amount (length) of token embeddings</li>
 * <li><b>layers</b> <i>(int)</i> - the amount of encoder blocks (stacks or layers)</li>
 * <li><b>load_model</b> <i>(boolean)</i> - if the model should be loaded from the file</li>
 * <li><b>learning_rate</b> <i>(float)</i> - model's learning rate</li>
 * <li><b>load_updater</b> <i>(boolean)</i> - if the model's updater data should be loaded from the file</li>
 * <li><b>prediction_masking_percentage</b> <i>(int)</i> - the percentage of the prediction tokens which should be masked. The model
 * could have theoretically each of the tokens masked, but the original BERT paper showed that using some of the tokens as not masked helps to
 * get better results</li>
 * <li><b>only_article_summaries</b> <i>(boolean)</i> - if only Wikipedia's article summaries should be fetched instead of all
 * paragraphs. This one is specific to {@link WikiArticlesContentProvider}</li>
 * <li><b>encoder_type</b> <i>(String)</i> - should be either "CLASSIC" or "COMPETENCE_BASED". Refer to {@link EncoderType} in
 * order to see all possible values</li>
 * <li><b>use_pos_tags</b> <i>(boolean)</i> - if POS-Tags should be used during training. This allows the model to learn and store
 * in the embeddings also the info regarding the part of the speech of each token. It might allow for better generalization and
 * attention normalization</li>
 * <li><b>MAX_JVM_MEMORY</b> <i>(String)</i> - the max amount of RAM which could be used by the model during training or inference,
 * e.g. "30G" </li>
 * <li><b>DEALLOCATOR_THREADS</b> <i>(int)</i> - amount of processor threads which could be used by for memory deallocation</li>
 * </ul>
 */
public class TrainMlm extends CommonTrainer {
    private static final Logger LOG = LoggerFactory.getLogger(TrainMlm.class);
    private static final Path ROOT_PATH = Paths.get(getenv().getOrDefault("root_path", "."));
    private static final Path MODEL_PATH = ROOT_PATH.resolve("wiki_mlm_encoder_model");
    private static final Path BACKUP_PATH = ROOT_PATH.resolve("wiki_mlm_encoder_model_backup");
    private static final Path LAST_PAGE_INDEX_FILE_PATH = ROOT_PATH.resolve("last_processed_wiki_page.txt");
    private static final int BATCH_SIZE = parseInt(getenv().getOrDefault("batch_size", "128"));
    private static final int LOG_FREQ = parseInt(getenv().getOrDefault("log_freq", "2"));
    private static final int TESTING_FREQ = parseInt(getenv().getOrDefault("test_freq", "100"));
    private static final int SAVE_FREQ = parseInt(getenv().getOrDefault("save_freq", "150"));
    private static final int MIN_SEQUENCE_UTILIZATION = parseInt(getenv().getOrDefault("min_sequence_utilization", "50"));
    private static final int EPOCHS = parseInt(getenv().getOrDefault("epochs", "1"));
    private static final int MAX_MEMORY_LOG_FREQ_MINUTES =
            parseInt(getenv().getOrDefault("max_memory_log_freq_minutes", "20"));
    private static final int ENCODER_LAYERS_AMOUNT = parseInt(getenv().getOrDefault("layers", "4"));
    private static final String TEST_DATA_FILE = getenv().get("test_data_file");
    private static final boolean LOAD_MODEL = parseBoolean(getenv().getOrDefault("load_model", "false"));
    private static final float LEARNING_RATE = parseFloat(getenv().getOrDefault("learning_rate", "0.0001"));
    private static final boolean LOAD_UPDATER = parseBoolean(System.getenv().getOrDefault("load_updater", "true"));
    private static final int PERCENTAGE_OF_PREDICTION_MASKING = parseInt(getenv().getOrDefault("prediction_masking_percentage", "80"));
    private static final int DIMENSIONS = parseInt(getenv().getOrDefault("dimensions", "768"));
    private static final boolean FETCH_ONLY_ARTICLE_SUMMARY = parseBoolean(getenv().getOrDefault("only_article_summaries", "false"));
    private static final String ENCODER_TYPE = getenv().getOrDefault("encoder_type", "CLASSIC");
    private static final boolean USE_POS_TAGS = parseBoolean(getenv().getOrDefault("use_pos_tags", "false"));

    private static final AtomicBoolean savingInProgress = new AtomicBoolean(false);

    public static void main(String[] args) throws IOException {
        prepareEnvironment();
        startMemoryProfiling(MAX_MEMORY_LOG_FREQ_MINUTES);
        int maxArticleIndex = 10000000;
        int articlesFetchStep = 20;
        int maxStepsAmount = 5000000;
        AtomicLong lastFetchedPageIndex = new AtomicLong(1);

        var mainVocab = loadVocab(getResourceFileStream("vocab.txt", MlmTransformerSdModel.class));
        var posTagsVocab = USE_POS_TAGS ? PosTagsVocab.loadVocab(getResourceFileStream("penn_treebank_pos_tags.txt",
                MlmTransformerSdModel.class)) : null;
        var transformer = getTransformerEncoderSdModel(mainVocab, posTagsVocab);

        if (!exists(LAST_PAGE_INDEX_FILE_PATH.toAbsolutePath())) {
            LOG.debug("Creating last page index file");
            createFile(LAST_PAGE_INDEX_FILE_PATH);
        }

        if (LOAD_MODEL) {
            loadModel(lastFetchedPageIndex, transformer);
        }

        LOG.info("Total model size is {} million params", format("%.2f", ((double) transformer.getTotalParamsAmount()) / 1000000));

        AtomicLong lastEpochLastFetchedPageIndex = new AtomicLong(lastFetchedPageIndex.get());

        try {
            addModelSaveSafeShutdownHook();

            WikiArticlesContentProvider wikiArticlesContentProvider =
                    getWikiArticlesContentProvider(lastFetchedPageIndex, articlesFetchStep);
            transformer.setDataProvider(wikiArticlesContentProvider);

            Map<String, String> testData = Optional.ofNullable(TEST_DATA_FILE)
                    .map(TrainMlm::loadTestData)
                    .orElse(new HashMap<>());
            LOG.debug("Starting training model");
            transformer.train(
                    (model, bufferFromLastEpoch) -> saveModel(model, lastEpochLastFetchedPageIndex.get()),
                    (model, bufferFromCurrentEpoch) -> saveModel(model, lastEpochLastFetchedPageIndex
                            .updateAndGet(current -> lastFetchedPageIndex.get() - bufferFromCurrentEpoch)),
                    maxStepsAmount,
                    transformerR -> transformerR.test(testData),
                    () -> lastFetchedPageIndex.get() >= maxArticleIndex);
            LOG.info("Completed the whole training");
            System.exit(0);
        } catch (Throwable t) {
            LOG.error("ERROR intercepted", t);
            // Explicitly calling exit in order to invoke the shutdown hook
            System.exit(-1);
        }
    }

    @Nonnull
    private static WikiArticlesContentProvider getWikiArticlesContentProvider(AtomicLong lastFetchedPageIndex, int articlesFetchStep) {
        Supplier<Collection<WikiTextArticle>> articlesSupplier = () -> {
            long startPageIndex = lastFetchedPageIndex.get();
            List<WikiTextArticle> wikiTextArticles = fetchArticlesFromDb(WikiTextArticle.class, articlesFetchStep,
                    gt("_id", lastFetchedPageIndex.get()));
            lastFetchedPageIndex.set(getLastPageId(wikiTextArticles));
            LOG.debug(" Fetching {} articles  - from {} till {}",
                    lastFetchedPageIndex.get() - startPageIndex, startPageIndex, lastFetchedPageIndex.get());
            return wikiTextArticles;
        };

        return new WikiArticlesContentProvider(articlesSupplier, FETCH_ONLY_ARTICLE_SUMMARY);
    }

    private static void loadModel(AtomicLong lastFetchedPageIndex, MlmTransformerSdModel transformer)
            throws IOException {
        loadModelVariablesAndUpdater(transformer, MODEL_PATH, BACKUP_PATH, LOAD_UPDATER);
        lastFetchedPageIndex.set(parseLong(
                Optional.of(readString(LAST_PAGE_INDEX_FILE_PATH).trim())
                        .filter(StringUtils::isNotBlank)
                        .orElse("0")));
    }

    private static MlmTransformerSdModel getTransformerEncoderSdModel(WordPieceVocab vocab, PosTagsVocab posTagsVocab) {
        var englishConjunctions = getResourceFileLines("english_conjunctors.txt", MlmTransformerSdModel.class)
                .stream()
                .map(String::toLowerCase)
                .collect(toList());


        Builder modelBuilder = new Builder(vocab, englishConjunctions, posTagsVocab)
                .withEncoderType(EncoderType.valueOf(ENCODER_TYPE))
                .withMiniEpochs(EPOCHS)
                .withSequenceLength(256)
                .withBatchSize(BATCH_SIZE)
                .withOwnLayersAmount(ENCODER_LAYERS_AMOUNT)
                .withSaveFrequencyInSteps(SAVE_FREQ)
                .withLoggingFrequency(LOG_FREQ)
                .withModelTestFrequency(TESTING_FREQ)
                .withMinimumSequenceUtilizationPercentage(MIN_SEQUENCE_UTILIZATION)
                .withAttentionHeadsAmount(8)
                .withLearningRate(LEARNING_RATE)
                .withHiddenSize(DIMENSIONS)
                .withIntermediateLayerSize((int) (DIMENSIONS * 1.334))
                .withPercentageOfMaskingPerPrediction(PERCENTAGE_OF_PREDICTION_MASKING)
                .withLabelSmoothing(0.1)
                .withMaxSizeOfWholePhrasePrediction(3);
        return modelBuilder.build();
    }

    private static Map<String, String> loadTestData(String fileName) {
        try {
            String testDataJson = readString(ROOT_PATH.resolve(fileName));
            return deserializeMapFromJsonWithJackson(testDataJson, String.class, String.class).orElseThrow();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private synchronized static void saveModel(MlmTransformerSdModel model, long startPageIndex) {
        savingInProgress.getAndSet(true);
        try {
            saveModel(model, MODEL_PATH, BACKUP_PATH);
            write(LAST_PAGE_INDEX_FILE_PATH, ImmutableSet.of(String.valueOf(startPageIndex)), TRUNCATE_EXISTING);
        } catch (Throwable t) {
            lastSaveWasSuccessful.set(false);
            LOG.error("Saving a model failed. ", t);
            try {
                throw t;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        } finally {
            savingInProgress.getAndSet(false);
        }
    }

    private static long getLastPageId(Collection<WikiTextArticle> pages) {
        return pages.stream()
                .mapToLong(WikiTextArticle::getId)
                .max()
                .orElse(0);
    }

}