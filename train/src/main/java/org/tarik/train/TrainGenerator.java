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
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.generational.GenerativeTransformerSdModel;
import org.tarik.core.network.models.transformer.generational.GenerativeTransformerSdModel.Builder;
import org.tarik.core.vocab.WordPieceVocab;
import org.tarik.train.db.model.wiki.WikiTextArticle;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Type;
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
import static java.lang.System.getenv;
import static java.nio.file.Files.*;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.util.stream.Collectors.toList;
import static org.bytedeco.javacpp.Pointer.physicalBytes;
import static org.tarik.core.vocab.WordPieceVocab.loadVocab;
import static org.tarik.train.db.model.wiki.WikiDatastore.fetchArticlesFromDb;
import static org.tarik.utils.ResourceLoadingUtil.getResourceFileLines;
import static org.tarik.utils.ResourceLoadingUtil.getResourceFileStream;

public class TrainGenerator extends CommonTrainer {
    private static final Logger LOG = LoggerFactory.getLogger(TrainGenerator.class);
    private static final Path ROOT_PATH = Paths.get(getenv().getOrDefault("root_path", "."));
    private static final Path MODEL_PATH = ROOT_PATH.resolve("wiki_generative_decoder_model");
    private static final Path BACKUP_PATH = ROOT_PATH.resolve("wiki_generative_decoder_model_backup");
    private static final Path LAST_PAGE_INDEX_FILE_PATH = ROOT_PATH.resolve("last_processed_gen_wiki_page.txt");
    private static final int BATCH_SIZE = parseInt(getenv().getOrDefault("batch_size", "128"));
    private static final int LOG_FREQ = parseInt(getenv().getOrDefault("log_freq", "2"));
    private static final int TESTING_FREQ = parseInt(getenv().getOrDefault("test_freq", "60"));
    private static final int SAVE_FREQ = parseInt(getenv().getOrDefault("save_freq", "100"));
    private static final int MIN_SEQUENCE_UTILIZATION = parseInt(getenv().getOrDefault("min_sequence_utilization", "50"));
    private static final int MAX_MEMORY_LOG_FREQ_MINUTES =
            parseInt(getenv().getOrDefault("max_memory_log_freq_minutes", "20"));
    private static final int ENCODER_LAYERS_AMOUNT = parseInt(getenv().getOrDefault("layers", "4"));
    private static final String TEST_DATA_FILE = getenv().get("generative_test_data_file");
    private static final boolean LOAD_MODEL = parseBoolean(getenv().getOrDefault("load_model", "false"));
    private static final float LEARNING_RATE = parseFloat(getenv().getOrDefault("learning_rate", "0.0001"));
    private static final boolean LOAD_UPDATER = parseBoolean(System.getenv().getOrDefault("load_updater", "true"));
    private static final int DIMENSIONS = parseInt(getenv().getOrDefault("dimensions", "768"));
    private static final boolean FETCH_ONLY_ARTICLE_SUMMARY = parseBoolean(getenv().getOrDefault("only_article_summaries", "false"));

    private static final AtomicBoolean savingInProgress = new AtomicBoolean(false);

    public static void main(String[] args) throws IOException {
        prepareEnvironment();
        startMemoryProfiling(MAX_MEMORY_LOG_FREQ_MINUTES);

        int maxArticleIndex = 10000000;
        int articlesFetchStep = 500;
        int maxStepsAmount = 5000000;
        AtomicLong lastFetchedPageIndex = new AtomicLong(1);

        var mainVocab = loadVocab(getResourceFileStream("vocab.txt", GenerativeTransformerSdModel.class));
        var transformer = getSdModel(mainVocab);

        if (!exists(LAST_PAGE_INDEX_FILE_PATH.toAbsolutePath())) {
            LOG.debug("Creating last page index file");
            createFile(LAST_PAGE_INDEX_FILE_PATH);
        }

        if (LOAD_MODEL) {
            loadModel(lastFetchedPageIndex, transformer);
        }

        AtomicLong lastEpochLastFetchedPageIndex = new AtomicLong(lastFetchedPageIndex.get());

        try {
            addModelSaveSafeShutdownHook();

            Supplier<Collection<WikiTextArticle>> articlesSupplier = () -> {
                long startPageIndex = lastFetchedPageIndex.get();
                List<WikiTextArticle> wikiTextArticles = fetchArticlesFromDb(WikiTextArticle.class, articlesFetchStep,
                        gt("_id", lastFetchedPageIndex.get()));
                lastFetchedPageIndex.set(getLastPageId(wikiTextArticles));
                LOG.debug(" Fetching {} articles  - from {} till {}",
                        lastFetchedPageIndex.get() - startPageIndex, startPageIndex, lastFetchedPageIndex.get());
                return wikiTextArticles;
            };

            var wikiArticlesContentProvider = new WikiArticlesContentProvider(articlesSupplier, FETCH_ONLY_ARTICLE_SUMMARY);
            transformer.setDataProvider(wikiArticlesContentProvider);

            var testData = Optional.ofNullable(TEST_DATA_FILE)
                    .map(TrainGenerator::loadTestData)
                    .orElse(new HashMap<>());
            LOG.debug("Starting model training");
            transformer.train(
                    (model, bufferFromLastEpoch) -> saveModel(model, lastEpochLastFetchedPageIndex.get()),
                    (model, bufferFromCurrentEpoch) -> saveModel(model, lastEpochLastFetchedPageIndex
                            .updateAndGet(current -> lastFetchedPageIndex.get() - bufferFromCurrentEpoch)),
                    maxStepsAmount,
                    t -> t.test(testData),
                    () -> lastFetchedPageIndex.get() >= maxArticleIndex);
            LOG.info("Completed the whole training");
            System.exit(0);
        } catch (Throwable t) {
            LOG.error("ERROR intercepted", t);
            LOG.warn("Memory taken: {} GB", (double) physicalBytes() / 1024 / 1024 / 1024);
            // Explicitly calling exit in order to invoke the shutdown hook
            System.exit(-1);
        }
    }

    private static void loadModel(AtomicLong lastFetchedPageIndex, GenerativeTransformerSdModel transformer)
            throws IOException {
        loadModelVariablesAndUpdater(transformer,MODEL_PATH, BACKUP_PATH,  LOAD_UPDATER);
        lastFetchedPageIndex.set(parseLong(
                Optional.of(readString(LAST_PAGE_INDEX_FILE_PATH).trim())
                        .filter(StringUtils::isNotBlank)
                        .orElse("0")));
    }

    private static GenerativeTransformerSdModel getSdModel(WordPieceVocab vocab) {
        var englishConjunctions = getResourceFileLines("english_conjunctors.txt", GenerativeTransformerSdModel.class)
                .stream()
                .map(String::toLowerCase)
                .collect(toList());

        var modelBuilder = new Builder(vocab, englishConjunctions)
                .withSequenceLength(256)
                .withPercentageOfTokensToBePredicted(15)
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
                .withLabelSmoothing(0.1);
        return modelBuilder.build();
    }

    private static Map<String[], String> loadTestData(String fileName) {
        try (var testDataStream = lines(ROOT_PATH.resolve(fileName))) {
            GsonBuilder gsonBuilder = new GsonBuilder();
            gsonBuilder.enableComplexMapKeySerialization();
            Type collectionType = new TypeToken<Map<String[], String>>() {
            }.getType();
            Gson gson = gsonBuilder.create();
            Map<String[], String> collector = new HashMap<>();
            testDataStream.map(string -> (Map<String[], String>) gson.fromJson(string, collectionType))
                    .forEach(collector::putAll);
            LOG.info("Loaded {} <sentence----masked word> test data pairs", collector.size());
            return collector;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private synchronized static void saveModel(GenerativeTransformerSdModel model, long startPageIndex) {
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