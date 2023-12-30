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

package org.tarik.core.network.models.transformer.mlm;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.layers.sd.transformer.TransformerEncoderGraphGenerator;
import org.tarik.core.network.layers.sd.transformer.TransformerExpertBasedEncoderGraphGenerator;
import org.tarik.core.network.models.transformer.AbstractOpenDatasetTransformerModel;
import org.tarik.core.network.models.transformer.AbstractTransformerSameDiffModel;
import org.tarik.core.vocab.PosTagsVocab;
import org.tarik.core.vocab.WordPieceVocab;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.io.Serial;
import java.io.UncheckedIOException;
import java.time.Instant;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.*;
import static java.lang.String.format;
import static java.time.Instant.now;
import static java.util.Arrays.fill;
import static java.util.Collections.sort;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.empty;
import static java.util.Optional.ofNullable;
import static java.util.concurrent.Executors.newSingleThreadExecutor;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.*;
import static java.util.stream.IntStream.generate;
import static java.util.stream.IntStream.range;
import static org.nd4j.common.base.Preconditions.checkArgument;
import static org.nd4j.linalg.api.buffer.DataType.*;
import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.indexing.BooleanIndexing.replaceWhere;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.tarik.core.network.models.transformer.mlm.MlmTransformerSdModel.EncoderType.CLASSIC;
import static org.tarik.core.parsing_utils.PosTagger.getPosTagsStream;
import static org.tarik.core.parsing_utils.WordSplitter.breakCorpusIntoWords;
import static org.tarik.utils.CommonUtils.getCounterEnding;
import static org.tarik.utils.CommonUtils.getDurationInSeconds;
import static org.tarik.utils.SdUtils.*;

/**
 * <p>
 * Concrete implementation of {@link AbstractOpenDatasetTransformerModel} aimed at performing the masked language modeling. This is a
 * BERT-like model which is based on the <a href="https://arxiv.org/pdf/1810.04805.pdf">original paper</a>.
 * This model could be based on two different types of encoders : the classic {@link TransformerEncoderGraphGenerator} and the
 * competence (expert) - based {@link TransformerExpertBasedEncoderGraphGenerator}.
 * </p>
 * <p>
 * The model is created using the builder pattern. In order to start the model's training,
 * {@link MlmTransformerSdModel#train(BiConsumer, BiConsumer, int, Consumer, Supplier)} method must be called. In order to evaluate the
 * model's accuracy, {@link MlmTransformerSdModel#test(Map)} method must be called. The resulting accuracy will be logged.
 * </p>
 * <p>
 * The model uses a custom {@link CustomListener} which allows to log different debug and execution data in order to show the
 * current training progress. It's also responsible for plotting the accuracy results.
 * </p>
 */
public class MlmTransformerSdModel extends AbstractOpenDatasetTransformerModel<MlmTransformerSdModel,
        MlmTransformerSdModel.PassageTokensSequenceTagged> {
    @Serial
    private static final long serialVersionUID = 1354020103435117457L;
    private static final String MODEL_FILE_NAME = "Transformer Masked Language Model";
    private static final Logger LOG = LoggerFactory.getLogger(MlmTransformerSdModel.class);
    public static final String ENCODER_ID = "encoder";
    protected static final String FLAT_PREDICTION_TOKEN_VOCAB_INDICES_VAR_NAME = "flatPredictionTokenVocabIndices";
    protected static final String INPUT_VAR_NAME = "inputTokenVocabIndices";
    protected static final String FLAT_PREDICTION_TOKEN_BATCH_POSITIONS_VAR_NAME = "flatPredictionTokenBatchPositions";
    protected static final String POSITIONAL_ATTENTION_MASKS_VAR_NAME = "positionalAttentionMasks";
    protected static final String TOKEN_EMBEDDINGS_MATRIX_VAR_NAME = "tokenEmbeddingsMatrix";
    protected static final String POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME = "positionalEmbeddingsMatrix";
    protected static final String FEATURE_MASKS_VAR_NAME = "inputMasks";
    protected static final String POS_TAG_LABELS_VAR_NAME = "posTagLabels";
    public static final String MODEL_OUTPUT_VAR_NAME = "finalLayerResult";
    public static final String POS_TAG_PREDICTION_LOGITS_VAR_NAME = "posTagPredictionLogits";
    private static final int MIN_SEQUENCE_WORD_TOKENS_AMOUNT = 20;

    private static final Random random = new Random();
    private final PosTagsVocab posTagsVocab;
    private final int percentageOfTokensToBePredicted;
    private final int maxSizeOfWholePhrasePrediction;
    private final int percentageOfMaskingPerPrediction;
    private final int miniEpochsAmount;
    private final EncoderType encoderType;

    public enum EncoderType {
        CLASSIC, COMPETENCE_BASED
    }

    protected MlmTransformerSdModel(double beta2, int hiddenSize, List<String> englishConjunctors, int intermediateLayerSize,
                                    double learningRate,
                                    double labelSmoothing, int attentionHeadsAmount, int encoderLayersAmount,
                                    int percentageOfTokensToBePredicted, int maxSizeOfWholePhrasePrediction,
                                    int percentageOfMaskingPerPrediction, int sequenceLength, int batchSize, int miniEpochsAmount,
                                    int modelSaveFrequency, WordPieceVocab vocab, int loggingFrequency, int testingFrequency,
                                    BertWordPieceTokenizerFactory tokenizerFactory, int minimumSequenceUtilizationPercentage,
                                    PosTagsVocab posTagsVocab, EncoderType encoderType) {
        super(batchSize, sequenceLength, testingFrequency, loggingFrequency, learningRate, hiddenSize, encoderLayersAmount,
                attentionHeadsAmount, intermediateLayerSize, labelSmoothing, false, false, modelSaveFrequency,
                beta2, 0f, englishConjunctors, minimumSequenceUtilizationPercentage, sequenceLength - 1);
        this.encoderType = encoderType;
        this.vocab = requireNonNull(vocab);
        this.percentageOfTokensToBePredicted = percentageOfTokensToBePredicted;
        this.maxSizeOfWholePhrasePrediction = maxSizeOfWholePhrasePrediction;
        this.percentageOfMaskingPerPrediction = percentageOfMaskingPerPrediction;
        this.miniEpochsAmount = miniEpochsAmount;
        this.tokenizerFactory = tokenizerFactory;
        this.posTagsVocab = posTagsVocab;

        buildCompleteSdGraph();

        LOG.info(""" 
                Created a transformer's encoder-based MLM model with the following params:
                -----------------------------------------------------------------------------
                {}
                -----------------------------------------------------------------------------
                """, this);
    }

    @Override
    protected TrainingConfig getTrainingConfig() {
        var dataSetLabelMapping = new LinkedList<>(List.of(FLAT_PREDICTION_TOKEN_VOCAB_INDICES_VAR_NAME,
                FLAT_PREDICTION_TOKEN_BATCH_POSITIONS_VAR_NAME));
        if (posTagsVocab != null) {
            dataSetLabelMapping.add(POS_TAG_LABELS_VAR_NAME);
        }

        return new TrainingConfig.Builder()
                .updater(getUpdater())
                .dataSetFeatureMapping(INPUT_VAR_NAME)
                .dataSetLabelMapping(dataSetLabelMapping)
                .dataSetFeatureMaskMapping(FEATURE_MASKS_VAR_NAME)
                .dataSetLabelMaskMapping(POSITIONAL_ATTENTION_MASKS_VAR_NAME)
                .build();
    }

    @Override
    protected String getModelFileName() {
        return MODEL_FILE_NAME;
    }

    @Override
    protected MlmTransformerSdModel getInstance() {
        return this;
    }

    @Override
    protected void buildCompleteSdGraph() {
        var sd = SameDiff.create();
        sd.setTrainingConfig(getTrainingConfig());
        //sd.enableDebugMode();
        sd.setLogExecution(false);
        sd.setEnableCache(false);

        LOG.info("""                            
                Building a transformer's encoder SameDiff model with the following params:
                 - number of encoder blocks: {}
                 - max sequence length: {}
                 - token embeddings and hidden layer size: {}
                 - vocab size: {}
                 - attention heads amount: {}
                """, layersAmount, baseSequenceLength, hiddenSize, vocab.size(), attentionHeadsAmount);

        // Inputs
        var inputMasks = sd.placeHolder(FEATURE_MASKS_VAR_NAME, FLOAT, -1, baseSequenceLength);
        var inputTokenVocabIndices = sd.placeHolder(INPUT_VAR_NAME, INT32, -1, baseSequenceLength);
        var positionalAttentionMasks =
                sd.placeHolder(POSITIONAL_ATTENTION_MASKS_VAR_NAME, FLOAT, -1, baseSequenceLength, baseSequenceLength);
        // flatPredictionTokenBatchPositions is a vector which contains the batch position indices of
        // the tokens which need to be predicted for the whole batch
        var flatPredictionTokenBatchPositions = sd.placeHolder(FLAT_PREDICTION_TOKEN_BATCH_POSITIONS_VAR_NAME, INT32, -1);

        // Embeddings
        var positionalEmbeddingsMatrix = sd.var(POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME, initializePositionalEmbeddings(baseSequenceLength))
                .convertToConstant();
        var tokenEmbeddingsMatrix = sd.var(TOKEN_EMBEDDINGS_MATRIX_VAR_NAME,
                new XavierInitScheme('c', vocab.size(), hiddenSize), FLOAT, vocab.size(), hiddenSize);
        var batchTokenEmbeddings = gather(sd, "batchTokenEmbeddings", tokenEmbeddingsMatrix, inputTokenVocabIndices, 0);

        //var transformerEncoderGraphGenerator = getTransformerSummarizerEncoderGraphGenerator();
        var transformerEncoderGraphGenerator = getTransformerEncoderGraphGenerator();

        var encoderOutput =
                transformerEncoderGraphGenerator.generateGraph(sd, inputMasks, batchTokenEmbeddings, positionalEmbeddingsMatrix);

        // Collecting final word embeddings for the positions which need to be predicted
        var predictionTokenEmbeddings =
                gather(sd, "predictionTokenEmbeddings", encoderOutput, flatPredictionTokenBatchPositions, 0);
        var flatPredictionTokenVocabIndices = sd.placeHolder(FLAT_PREDICTION_TOKEN_VOCAB_INDICES_VAR_NAME, INT32, -1);
        var outputLayerBias = sd.zero("outputLayerBias", vocab.size()).convertToVariable();

        // Calculating POS tagging loss
        var flatBatchSize = getDimensionSize("flatBatchSize", encoderOutput, 0);
        if (posTagsVocab != null) {
            definePosTaggingLoss(sd, flatPredictionTokenBatchPositions, flatBatchSize, predictionTokenEmbeddings);
        }

        // Calculating loss
        var outputLayerResult = sd.mmul(predictionTokenEmbeddings, tokenEmbeddingsMatrix, false, true, false)
                .add("outputLayerResult", outputLayerBias);
        var oneHotLabels = sd.oneHot(flatPredictionTokenVocabIndices, vocab.size());
        var oneHotSmoothLabels = oneHotLabels.mul(1 - labelSmoothing).plus(labelSmoothing / vocab.size());
        var logitPredictions = softmaxForLossCalculation(sd, MODEL_OUTPUT_VAR_NAME, outputLayerResult, 1, oneHotSmoothLabels);
        var loss = sd.math.log(logitPredictions).mul(oneHotLabels).sum(1).mean(0).neg("mainLoss");
        loss.markAsLoss();

        this.sameDiff = sd;
    }

    private void definePosTaggingLoss(SameDiff sd, SDVariable flatPredictionBatchPositions, SDVariable flatBatchSize,
                                      SDVariable predictionTokenEmbeddings) {
        var posTagClassifierWeights = sd.var("posTagClassifierWeights",
                new XavierInitScheme('c', hiddenSize, posTagsVocab.size()), FLOAT, hiddenSize, posTagsVocab.size());
        var posTagClassifierBias = sd.zero("posTagClassifierBias", posTagsVocab.size()).convertToVariable();
        var flatPosTagLabels = sd.placeHolder(POS_TAG_LABELS_VAR_NAME, FLOAT, -1, baseSequenceLength)
                .reshape(flatBatchSize).rename("flatPosTagLabels");
        var predictionTokenPosTagLabels =
                gather(sd, "predictionTokenPosTagLabels", flatPosTagLabels, flatPredictionBatchPositions, 0);
        var posTagPredictions =
                sd.nn().linear("posTagPredictions", predictionTokenEmbeddings, posTagClassifierWeights, posTagClassifierBias);
        var oneHotLabels = sd.oneHot(predictionTokenPosTagLabels, posTagsVocab.size());
        var logitPredictions = softmaxForLossCalculation(sd, POS_TAG_PREDICTION_LOGITS_VAR_NAME, posTagPredictions, 1, oneHotLabels);
        sd.math.log(logitPredictions).mul(oneHotLabels).sum(1).mean(0).neg("posTagPredictionLoss").markAsLoss();
    }

    private TransformerEncoderGraphGenerator getTransformerEncoderGraphGenerator() {
        return switch (encoderType) {
            case CLASSIC -> getClassicalEncoderGraphGenerator();
            case COMPETENCE_BASED -> getTransformerCompetenceBasedEncoderGraphGenerator();
        };
    }

    @Nonnull
    private TransformerEncoderGraphGenerator getTransformerCompetenceBasedEncoderGraphGenerator() {
        return new TransformerExpertBasedEncoderGraphGenerator.Builder(ENCODER_ID)
                .withAttentionHeadsAmount(this.attentionHeadsAmount)
                .withLayersAmount(this.layersAmount)
                .withHiddenSize(hiddenSize)
                .withIntermediateLayerSize(this.intermediateLayerSize)
                .withSequenceLength(this.baseSequenceLength)
                .build();
    }

    @Nonnull
    private TransformerEncoderGraphGenerator getClassicalEncoderGraphGenerator() {
        return new TransformerEncoderGraphGenerator.Builder(ENCODER_ID)
                .withAttentionHeadsAmount(this.attentionHeadsAmount)
                .withLayersAmount(this.layersAmount)
                .withHiddenSize(hiddenSize)
                .withIntermediateLayerSize(this.intermediateLayerSize)
                .withSequenceLength(this.baseSequenceLength)
                .build();
    }

    public void train(BiConsumer<MlmTransformerSdModel, Long> incompleteEpochModelSaver,
                      BiConsumer<MlmTransformerSdModel, Long> completeEpochModelSaver,
                      int totalStepsAmount, Consumer<MlmTransformerSdModel> tester,
                      Supplier<Boolean> limitReached)
            throws IOException, ExecutionException, InterruptedException {
        requireNonNull(dataProvider, "Model can't be trained without a data provider");
        int processedStepsAmount = sameDiff.getTrainingConfig().getIterationCount();
        ExecutorService modelSaveExecutor = newSingleThreadExecutor();

        // Listeners
        CustomListener customListener = new CustomListener(loggingFrequency);
        sameDiff.setListeners(customListener);

        LinkedList<PassageTokensSequenceTagged> remainingTokenSentences = new LinkedList<>();

        // Breaking the provided data into batches and training for each available batch
        while (processedStepsAmount <= totalStepsAmount) {
            if ((processedStepsAmount + 1) % 50 == 0) {
                LOG.info("Starting {} training step having a buffer of {} passages from previous training",
                        getCounterEnding(processedStepsAmount + 1), remainingTokenSentences.size());
            }

            var batchedTokenSentences = getNewTokenBatch(remainingTokenSentences,
                    fetchedBlocks -> fetchedBlocks.size() > batchSize || limitReached.get());
            long categoriesBufferFromCurrentMiniEpoch = remainingTokenSentences.size();

            if (batchedTokenSentences.isEmpty()) {
                // Data provider is empty - can't continue
                LOG.warn("Interrupting training after {} steps because no more train data is available", processedStepsAmount);
                break;
            }

            // Additional logging info
            addLogInfo(batchedTokenSentences);

            // For each iteration the training data set will be different - so that predictions and labels are not
            // the same for each iteration. This allows to avoid fixing the network's attention on the same sequence positions
            Map<Integer, Set<TokenUsageStats>> collectedTokenUsageStats = new HashMap<>();
            for (int i = 0; i < miniEpochsAmount && processedStepsAmount <= totalStepsAmount; i++) {
                var batchTokenVocabIndices = create(INT32, batchedTokenSentences.size(), baseSequenceLength);
                List<Integer> flatPredictionTokenVocabIndices = new LinkedList<>();
                var batchFeatureMasks = create(INT8, batchedTokenSentences.size(), baseSequenceLength);
                List<Integer> flatPredictionTokenBatchPositions = new LinkedList<>();
                var batchPositionalAttentionMasks = create(INT8, batchedTokenSentences.size(), baseSequenceLength, baseSequenceLength);
                var posTagLabels = posTagsVocab == null ? null : create(INT8, batchedTokenSentences.size(), baseSequenceLength);

                range(0, batchedTokenSentences.size())
                        .filter(batchPosition -> batchedTokenSentences.get(batchPosition).getWordPieceTokens().size() > 40)
                        .forEach(batchPosition -> addSequenceTargetsAndLabels(batchPosition, batchedTokenSentences.get(batchPosition),
                                flatPredictionTokenVocabIndices, batchTokenVocabIndices, batchFeatureMasks, collectedTokenUsageStats,
                                flatPredictionTokenBatchPositions, batchPositionalAttentionMasks, posTagLabels));

                var multiDataSet = getMultiDataSet(batchTokenVocabIndices, flatPredictionTokenVocabIndices,
                        flatPredictionTokenBatchPositions, batchFeatureMasks, batchPositionalAttentionMasks, posTagLabels);
                sameDiff.fit(new SingletonMultiDataSetIterator(multiDataSet), 1);
                ++processedStepsAmount;

                System.gc();

                if (customListener.earlyTrainingStop) {
                    LOG.warn("Interrupting the training at {} step because early stop was required",
                            getCounterEnding(processedStepsAmount));
                    customListener.earlyTrainingStop = false;
                    break;
                }

                if (processedStepsAmount % modelTestFrequency == 0) {
                    tester.accept(this);
                    sameDiff.getSessions().clear();
                    System.gc();
                }
            }

            if (processedStepsAmount % saveFrequencyInSteps == 0) {
                LOG.info("Saving a model after {} step", getCounterEnding(processedStepsAmount));
                completeEpochModelSaver.accept(this, categoriesBufferFromCurrentMiniEpoch);
            }
        }
        LOG.info("Saving a model at the end of training");
        completeEpochModelSaver.accept(this, 0L);
        LOG.info("Completed training after {} steps", processedStepsAmount);
    }

    public void test(Map<String, String> maskedWordsBySentence) {
        Instant start = now();
        List<Double> accuracies = new LinkedList<>();
        LinkedList<Entry<String, String>> maskedWordsBySentenceQueue =
                new LinkedList<>(maskedWordsBySentence.entrySet());

        while (!maskedWordsBySentenceQueue.isEmpty()) {
            Map<String, String> batchMaskedWordsBySentence = new HashMap<>();
            LinkedList<Pair<List<String>, LinkedList<Integer>>> predictionTokenPositionsBySentenceTokens =
                    new LinkedList<>();
            while (batchMaskedWordsBySentence.size() < batchSize && !maskedWordsBySentenceQueue.isEmpty()) {
                Entry<String, String> maskedWordBySentence = maskedWordsBySentenceQueue.poll();
                batchMaskedWordsBySentence.put(maskedWordBySentence.getKey(), maskedWordBySentence.getValue());
            }

            batchMaskedWordsBySentence.forEach((testSentence, maskedWord) -> {
                List<String> words = new LinkedList<>(breakCorpusIntoWords(testSentence));
                int predictionWordPosition = words.indexOf(maskedWord);
                if (predictionWordPosition < baseSequenceLength - 1) {
                    LinkedList<String> wordPieceTokensBeforePredictionPosition =
                            words.subList(0, predictionWordPosition).stream()
                                    .map(sentence -> tokenizerFactory.create(sentence).getTokens())
                                    .flatMap(Collection::stream)
                                    .collect(toCollection(LinkedList::new));
                    wordPieceTokensBeforePredictionPosition.addFirst(WordPieceVocab.CLASSIFICATION_SYMBOL);
                    List<String> predictionTokens = tokenizerFactory.create(maskedWord).getTokens();
                    List<String> wordPieceTokensAfterPredictionPosition =
                            words.subList(predictionWordPosition + 1, words.size()).stream()
                                    .map(sentence -> tokenizerFactory.create(sentence).getTokens())
                                    .flatMap(Collection::stream)
                                    .collect(toList());

                    if (!predictionTokens.isEmpty()) {
                        LinkedList<Integer> predictionTokenPositions = range(wordPieceTokensBeforePredictionPosition.size(),
                                wordPieceTokensBeforePredictionPosition.size() + predictionTokens.size())
                                .boxed()
                                .collect(toCollection(LinkedList::new));
                        List<String> allSequenceTokens = Stream.of(wordPieceTokensBeforePredictionPosition, predictionTokens,
                                        wordPieceTokensAfterPredictionPosition)
                                .flatMap(Collection::stream)
                                .collect(toCollection(LinkedList::new));

                        predictionTokenPositionsBySentenceTokens.add(Pair.of(allSequenceTokens, predictionTokenPositions));
                    }
                }
            });

            while (!predictionTokenPositionsBySentenceTokens.isEmpty()) {
                Map<Integer, String> predictionTokenByBatchIndex = new HashMap<>();
                INDArray batchTokenVocabIndices = create(INT32, predictionTokenPositionsBySentenceTokens.size(), baseSequenceLength);
                INDArray batchFeatureMasks = create(INT8, predictionTokenPositionsBySentenceTokens.size(), baseSequenceLength);
                List<Integer> flatPredictionTokenBatchPositions = new LinkedList<>();
                AtomicInteger currentIndex = new AtomicInteger();
                predictionTokenPositionsBySentenceTokens.forEach(predictionPositionsBySentenceTokens -> {
                    int predictionTokenPosition = predictionPositionsBySentenceTokens.getValue().pollFirst();
                    LinkedList<String> allSequenceTokens = new LinkedList<>(predictionPositionsBySentenceTokens.getKey());
                    String predictionToken = allSequenceTokens.set(predictionTokenPosition, WordPieceVocab.MASK_SYMBOL);
                    int[] sentenceTokenIndices = allSequenceTokens.stream()
                            .mapToInt(token -> vocab.getTokenIndex(token).orElseThrow())
                            .toArray();
                    int[] sequenceFeatureMasks = generate(() -> 1).limit(baseSequenceLength).toArray();
                    sequenceFeatureMasks[predictionTokenPosition] = 0;
                    int[] wholeSequenceTokenIndices = Arrays.copyOf(sentenceTokenIndices, baseSequenceLength);
                    if (sentenceTokenIndices.length < baseSequenceLength) {
                        fill(sequenceFeatureMasks, sentenceTokenIndices.length, baseSequenceLength, 0);
                        fill(wholeSequenceTokenIndices, sentenceTokenIndices.length, baseSequenceLength,
                                vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow());
                    }
                    predictionTokenByBatchIndex.put(currentIndex.get(), predictionToken);
                    batchTokenVocabIndices.putRow(currentIndex.get(), createFromArray(wholeSequenceTokenIndices));
                    batchFeatureMasks.putRow(currentIndex.get(), createFromArray(sequenceFeatureMasks));
                    flatPredictionTokenBatchPositions.add(currentIndex.get() * baseSequenceLength + predictionTokenPosition);
                    currentIndex.incrementAndGet();
                });

                var predictedTokens = predict(batchTokenVocabIndices, batchFeatureMasks, flatPredictionTokenBatchPositions, 3);
                range(0, predictedTokens.size())
                        .forEach(i -> {
                            Map<String, Integer> predictionsByPosition = range(0, predictedTokens.get(i).size())
                                    .boxed()
                                    .collect(toMap(predictedTokens.get(i)::get, identity()));
                            double accuracy = ofNullable(predictionsByPosition.get(predictionTokenByBatchIndex.get(i)))
                                    .map(actualPredictionPosition -> 100 / ((double) actualPredictionPosition + 1))
                                    .orElse(0d);
                            accuracies.add(accuracy);
                        });

                predictionTokenPositionsBySentenceTokens.removeIf(predictionPositionsBySentenceTokens ->
                        predictionPositionsBySentenceTokens.getValue().isEmpty());
            }
        }

        double accuracyAverage = accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(-1);
        LOG.info("\n\nAVERAGE ACCURACY : {}%\n\n", format("%.2f", accuracyAverage));
        LOG.info("Inference took {} sec", getDurationInSeconds(start));
    }

    @Override
    protected synchronized void fetchNewDataBlock(List<PassageTokensSequenceTagged> collector,
                                                  Function<List<String>, Boolean> isLimitReachedFunction) {
        var sentencesQueue = new LinkedList<>(dataProvider.getPassages(isLimitReachedFunction));
        if (!sentencesQueue.isEmpty()) {
            while (!sentencesQueue.isEmpty()) {
                var passageTokensSequence = generateTruncatedPassageTokensSequence(sentencesQueue);
                if (passageTokensSequence.posTags.size() >= baseSequenceLength * minimumSequenceUtilizationPercentage / 100) {
                    collector.add(passageTokensSequence);
                }
            }

            if (collector.size() < batchSize) {
                // Recursive collection till the required amount is reached
                fetchNewDataBlock(collector, isLimitReachedFunction);
            }
        }
    }

    protected List<PassageTokensSequenceTagged> getNewTokenBatch(LinkedList<PassageTokensSequenceTagged> collector,
                                                                 Function<List<String>, Boolean> isLimitReachedFunction) {
        List<PassageTokensSequenceTagged> batchedPassageTokenSequences = new LinkedList<>();
        // Fetching only if there's still something to fetch
        if (collector.isEmpty()) {
            fetchNewDataBlock(collector, isLimitReachedFunction);
        }

        while (batchedPassageTokenSequences.size() < batchSize && !collector.isEmpty()) {
            var tokensSequence = collector.pollFirst();
            if (collector.isEmpty()) {
                fetchNewDataBlock(collector, isLimitReachedFunction);
            }
            if (tokensSequence.getWordPieceTokens().stream()
                    .filter(this::isValidToken)
                    .count() > MIN_SEQUENCE_WORD_TOKENS_AMOUNT) {
                batchedPassageTokenSequences.add(tokensSequence);
            }
        }
        return batchedPassageTokenSequences;
    }

    @Override
    protected PassageTokensSequenceTagged generateTruncatedPassageTokensSequence(LinkedList<String> sentencesQueue) {
        List<String> wordPieceTokens = new LinkedList<>();
        List<String> tokenPosTags = new LinkedList<>();
        while (!sentencesQueue.isEmpty() && wordPieceTokens.size() < totalSequenceCapacity) {
            var sentence = sentencesQueue.pollFirst();
            var fullWords = breakCorpusIntoWords(sentence);
            List<String> posTags = getPosTagsStream(sentence).toList();
            if (fullWords.size() == posTags.size()) {
                List<Pair<String, List<String>>> tokensByPosTagList = new LinkedList<>();
                range(0, fullWords.size()).forEach(index ->
                        tokensByPosTagList.add(Pair.of(posTags.get(index), tokenizerFactory.create(fullWords.get(index)).getTokens())));
                // Adding new sentence tokens only if there's still place left in the sequence
                if (tokensByPosTagList.stream().map(Pair::getValue).mapToLong(Collection::size).sum() <
                        totalSequenceCapacity - wordPieceTokens.size()) {
                    tokensByPosTagList.forEach(tokensByPosTag -> tokensByPosTag.getValue().forEach(token -> {
                        wordPieceTokens.add(token);
                        tokenPosTags.add(tokensByPosTag.getKey());
                    }));
                } else {
                    break;
                }
            } else {
                LOG.warn("""
                        Got different word/POST tag lengths.
                        Words:{}
                        POS Tags: {}""", fullWords, posTags);
            }
        }

        return new PassageTokensSequenceTagged(wordPieceTokens, tokenPosTags);
    }


    private MultiDataSet getMultiDataSet(INDArray batchTokenVocabIndices, List<Integer> flatPredictionTokenVocabIndices,
                                         List<Integer> flatPredictionTokenBatchPositions, INDArray batchFeatureMasks,
                                         INDArray batchPositionalAttentionMasks, @Nullable INDArray posTagLabels) {
        var labels = new LinkedList<>(List.of(createFromArray(flatPredictionTokenVocabIndices.toArray(Integer[]::new)),
                createFromArray(flatPredictionTokenBatchPositions.toArray(Integer[]::new))));
        var labelMasks = new LinkedList<>(List.of(batchPositionalAttentionMasks, Nd4j.empty()));
        if (posTagLabels != null) {
            labels.add(posTagLabels);
            labelMasks.add(Nd4j.empty());
        }

        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{batchTokenVocabIndices},
                labels.toArray(INDArray[]::new),
                new INDArray[]{batchFeatureMasks},
                labelMasks.toArray(INDArray[]::new));
    }

    private void addSequenceTargetsAndLabels(int batchPosition, PassageTokensSequenceTagged passageTokensSequenceTagged,
                                             List<Integer> flatPredictionTokenVocabIndices,
                                             INDArray batchTokenVocabIndices, INDArray batchFeatureMasks,
                                             Map<Integer, Set<TokenUsageStats>> collectedTokenUsageStatsByBatchPosition,
                                             List<Integer> flatPredictionTokenBatchPositions,
                                             INDArray batchPositionalAttentionMasks, @Nullable INDArray posTagLabels) {
        collectedTokenUsageStatsByBatchPosition.putIfAbsent(batchPosition, new HashSet<>());
        if (posTagLabels != null) {
            posTagLabels.putRow(batchPosition, getSequencePosTagLabels(passageTokensSequenceTagged));
        }
        var tokens = passageTokensSequenceTagged.getWordPieceTokens();

        int expectedPredictionsAmount = max(1, (tokens.size() * percentageOfTokensToBePredicted) / 100);
        try {
            var predictionData = generateSequenceRandomPredictionData(expectedPredictionsAmount, tokens,
                    collectedTokenUsageStatsByBatchPosition.get(batchPosition), 1);
            Set<Integer> maskedPredictionPositions = getMaskedPredictionPositions(predictionData);
            checkArgument(!maskedPredictionPositions.isEmpty(), "There must be at least 1 masked position");
            maskedPredictionPositions.add(0);
            addSequenceFeaturesAndLabels(batchPosition, flatPredictionTokenVocabIndices, batchTokenVocabIndices,
                    flatPredictionTokenBatchPositions, tokens, predictionData.predictionPositions(), maskedPredictionPositions);

            var tokenAttentionMasks = getTokenSelfAttentionMasks(maskedPredictionPositions, tokens.size());
            batchFeatureMasks.putRow(batchPosition, tokenAttentionMasks);
            var sequencePositionalAttentionMasks = tile(expandDims(tokenAttentionMasks, 0), baseSequenceLength, 1);
            batchPositionalAttentionMasks.put(new INDArrayIndex[]{point(batchPosition)}, sequencePositionalAttentionMasks);
        } catch (Exception e) {
            LOG.error("Got an exception while processing the sequence : {}", passageTokensSequenceTagged);
            throw new IllegalStateException(e);
        }
    }

    private void addSequenceFeaturesAndLabels(int batchPosition, List<Integer> flatPredictionTokenVocabIndices,
                                              INDArray batchTokenVocabIndices, List<Integer> flatPredictionTokenBatchPositions,
                                              List<String> tokens, List<Integer> predictionPositions,
                                              Set<Integer> maskedPredictionPositions) {
        INDArray sequenceTokenVocabIndices = create(INT32, baseSequenceLength);

        // Processing CLS token, always the first token
        sequenceTokenVocabIndices.putScalar(0, vocab.getTokenIndex(WordPieceVocab.CLASSIFICATION_SYMBOL).orElseThrow());
        int remainingSequenceLength = baseSequenceLength - 1;

        // Processing other tokens
        range(0, tokens.size())
                .limit(remainingSequenceLength)
                .forEach(index -> addTokenData(batchPosition, tokens.get(index), flatPredictionTokenVocabIndices,
                        flatPredictionTokenBatchPositions,
                        sequenceTokenVocabIndices, predictionPositions, maskedPredictionPositions, index + 1));

        // Padding remaining sequence space, all paddings should be exempted from attention
        if (tokens.size() < remainingSequenceLength) {
            range(tokens.size(), remainingSequenceLength)
                    .forEach(index -> sequenceTokenVocabIndices.putScalar(index,
                            vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow()));
        }

        batchTokenVocabIndices.putRow(batchPosition, sequenceTokenVocabIndices);
    }

    private Set<Integer> getMaskedPredictionPositions(PredictionData predictionData) {
        checkArgument(!predictionData.predictionPositions().isEmpty(), "There must be at least 1 prediction position");
        List<Integer> maskingCandidates = predictionData.predictionPositions().stream()
                .filter(predictionPosition -> !predictionData.maskExclusions().contains(predictionPosition))
                .collect(toCollection(LinkedList::new));
        int availableMaskedPredictionsAmount = (predictionData.predictionPositions().size() - predictionData.maskExclusions().size()) *
                percentageOfMaskingPerPrediction / 100;

        return getMaskedPredictionPositions(max(1, availableMaskedPredictionsAmount), maskingCandidates);
    }

    private void addTokenData(int batchPosition, String token, List<Integer> flatPredictionTokenVocabIndices,
                              List<Integer> flatPredictionTokenBatchPositions, INDArray sequenceTokenVocabIndices,
                              List<Integer> predictionPositions, Set<Integer> maskedPredictionPositions, int tokenPosition) {
        vocab.getTokenIndex(token)
                .ifPresentOrElse(tokenVocabIndex -> {
                            int finalTokenVocabIndex = tokenVocabIndex;
                            if (predictionPositions.contains(tokenPosition)) {
                                // Adding prediction indices with offset
                                flatPredictionTokenVocabIndices.add(tokenVocabIndex);
                                flatPredictionTokenBatchPositions.add(batchPosition * baseSequenceLength + tokenPosition);
                                if (maskedPredictionPositions.contains(tokenPosition)) {
                                    finalTokenVocabIndex = vocab.getTokenIndex(WordPieceVocab.MASK_SYMBOL).orElseThrow();
                                }
                            }
                            sequenceTokenVocabIndices.putScalar(tokenPosition, finalTokenVocabIndex);
                        },
                        // Shouldn't happen anyway, byt in case the tokenizer uses a different vocab and passed unknown tokens through -
                        // they should be hidden from input
                        () -> sequenceTokenVocabIndices.putScalar(tokenPosition,
                                vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow())
                );
    }

    private INDArray getSequencePosTagLabels(PassageTokensSequenceTagged sequence) {
        INDArray sequencePosTagLabels = create(INT8, baseSequenceLength);
        List<Integer> tokenPosTagIndices = new LinkedList<>();

        // CLS token
        tokenPosTagIndices.add(posTagsVocab.getTagIndex(PosTagsVocab.PADDING_TAG).orElseThrow());

        // Remaining tokens
        sequence.posTags.forEach(posTag ->
                tokenPosTagIndices.add(
                        posTagsVocab.getTagIndex(posTag).orElse(posTagsVocab.getTagIndex(PosTagsVocab.UNKNOWN_TAG).orElseThrow())));

        range(0, tokenPosTagIndices.size())
                .limit(baseSequenceLength)
                .forEach(index -> sequencePosTagLabels.putScalar(index, tokenPosTagIndices.get(index)));

        // Padding the rest of the sequence
        for (int i = tokenPosTagIndices.size(); i < baseSequenceLength; i++) {
            sequencePosTagLabels.putScalar(i, posTagsVocab.getTagIndex(PosTagsVocab.PADDING_TAG).orElseThrow());
        }

        return sequencePosTagLabels;
    }

    /**
     * Generates the list of positions which will be predicted and the ones which should be excluded from masking.
     * Also, the positions will be chosen so that one phrase with a randomly chosen amount of words will be predicted.
     * Only one of those positions will be masked, others will be exempted from masking.
     * Prioritisation during the selection:
     * 1. All positions must be "normally" distributed among the sequence (no clustering)
     * 2. Positions which have not yet been predicted for this sequence will be chosen
     * 3. Finally, all positions which were least frequently predicted for this sequence will be chosen
     *
     * @param expectedPredictionsAmount expectedPredictionsAmount
     * @param sequence                  sequence
     * @param collectedTokenUsageStats  list of objects which represent the frequency of prediction of each token for
     *                                  this sequence
     * @return randomly selected positions for prediction
     */
    private PredictionData generateSequenceRandomPredictionData(int expectedPredictionsAmount, List<String> sequence,
                                                                Set<TokenUsageStats> collectedTokenUsageStats, int sequenceTokensShift) {
        int distanceBetweenPredictions = sequence.size() / expectedPredictionsAmount;
        // First token should never be used for prediction
        Map<Integer, String> candidateTokensByPosition = range(0, min(sequence.size(), baseSequenceLength - sequenceTokensShift))
                .filter(index -> isValidToken(sequence.get(index)))
                .boxed()
                .collect(toMap(identity(), sequence::get));


        Map<Integer, String> notYetKnownTokensByPosition = candidateTokensByPosition.entrySet().stream()
                .filter(sequenceTokenByPosition -> collectedTokenUsageStats.stream()
                        .noneMatch(usageStats -> usageStats.getToken().equals(sequenceTokenByPosition.getValue())))
                .collect(toMap(Entry::getKey, Entry::getValue));

        LinkedList<Integer> predictionPositions = new LinkedList<>();

        // First choosing randomly between not yet predicted in this epoch tokens, if any
        addNotYetPredictedTokenPositions(expectedPredictionsAmount, collectedTokenUsageStats, distanceBetweenPredictions,
                candidateTokensByPosition, notYetKnownTokensByPosition, predictionPositions);

        // If there's still some place for predictions - choosing the tokens which have the lowest prediction usage
        addLeastFrequentlyPredictedTokenPositions(expectedPredictionsAmount, collectedTokenUsageStats, distanceBetweenPredictions,
                candidateTokensByPosition, notYetKnownTokensByPosition, predictionPositions);

        // Modifying collected prediction positions so that the randomly chosen amount of words which form a single phrase is included.
        sort(predictionPositions);
        int amountOfWordsInPhrase = max(1, random.nextInt(maxSizeOfWholePhrasePrediction + 1));

        // Only the positions which can fit all tokens of the phrase and which contain no punctuation characters or
        // conjunction words are selected
        List<Integer> validPhrasePredictionPositions = predictionPositions.stream()
                .filter(position -> !sequence.get(position).startsWith(WordPieceVocab.WORD_CONJUNCTION_SYMBOL))
                .filter(position -> getPhraseLastTokenPosition(sequence, position, amountOfWordsInPhrase)
                        .map(lastTokenPosition -> sequence.subList(position, lastTokenPosition + 1)
                                .stream()
                                .noneMatch(predictionToken -> predictionToken.matches(PUNCTUATIONS_PATTERN) ||
                                        englishConjunctors.contains(predictionToken)))
                        .orElse(false))
                .toList();
        List<Integer> maskExclusions = new LinkedList<>();
        if (!validPhrasePredictionPositions.isEmpty()) {
            int phrasePredictionFirstPosition =
                    validPhrasePredictionPositions.get(random.nextInt(validPhrasePredictionPositions.size()));
            int amountOfPhraseTokens = getPhraseLastTokenPosition(sequence, phrasePredictionFirstPosition, amountOfWordsInPhrase)
                    .map(position -> position - phrasePredictionFirstPosition + 1)
                    .orElse(0);
            int phraseMaskingPosition = random.nextInt(amountOfPhraseTokens) + phrasePredictionFirstPosition;
            range(phrasePredictionFirstPosition, amountOfPhraseTokens + phrasePredictionFirstPosition)
                    .filter(position -> position != phraseMaskingPosition)
                    .boxed()
                    .forEach(maskExclusions::add);
            int phrasePredictionStartIndex = predictionPositions.indexOf(phrasePredictionFirstPosition);
            int currentPositionIndex = phrasePredictionFirstPosition;

            for (int i = phrasePredictionStartIndex; i < phrasePredictionStartIndex + amountOfPhraseTokens; i++) {
                if (i < predictionPositions.size()) {
                    // Overwrite already existing position by the position of the phrase token
                    int removedPosition = predictionPositions.set(i, currentPositionIndex++);
                    decrementTokenUsageStats(collectedTokenUsageStats, sequence.get(removedPosition));
                } else {
                    // Otherwise, add a new prediction position
                    predictionPositions.add(currentPositionIndex++);
                }
                addTokenUsageStats(collectedTokenUsageStats, sequence.get(currentPositionIndex - 1));
            }
        }

        var predictionPositionsShifted = predictionPositions.stream().map(i -> sequenceTokensShift + i).toList();
        var maskExclusionsShifted = maskExclusions.stream().map(i -> sequenceTokensShift + i).toList();

        return new PredictionData(predictionPositionsShifted, maskExclusionsShifted);
    }

    private boolean isValidToken(String token) {
        return token.length() > 1 && !token.matches(PUNCTUATIONS_PATTERN) && !englishConjunctors.contains(token);
    }

    private record PredictionData(List<Integer> predictionPositions, List<Integer> maskExclusions) {
    }

    private Optional<Integer> getPhraseLastTokenPosition(List<String> sequence, int phraseStartPosition,
                                                         int expectedAmountOfWordsInPhrase) {
        int numberOfProcessedWords = 0;
        int currentCursorPosition = phraseStartPosition;
        while (numberOfProcessedWords < expectedAmountOfWordsInPhrase && currentCursorPosition < sequence.size() - 1) {
            if (!sequence.get(currentCursorPosition).startsWith(WordPieceVocab.WORD_CONJUNCTION_SYMBOL)) {
                ++numberOfProcessedWords;
            }
            ++currentCursorPosition;
        }
        return numberOfProcessedWords == expectedAmountOfWordsInPhrase ? Optional.of(currentCursorPosition - 1) :
                empty();
    }

    private Set<Integer> getMaskedPredictionPositions(int expectedMaskedPredictionsAmount,
                                                      List<Integer> predictionPositions) {
        Set<Integer> maskedPredictionPositions = new HashSet<>();
        while (maskedPredictionPositions.size() < expectedMaskedPredictionsAmount) {
            maskedPredictionPositions.add(predictionPositions.get(random.nextInt(predictionPositions.size())));
        }
        return maskedPredictionPositions;
    }

    private List<List<String>> predict(INDArray batchTokenVocabIndices, INDArray batchFeatureMasks,
                                       List<Integer> flatPredictionTokenBatchPositions, int limit) {
        sameDiff.getVariable(INPUT_VAR_NAME).setArray(batchTokenVocabIndices);
        sameDiff.getVariable(FEATURE_MASKS_VAR_NAME).setArray(batchFeatureMasks);
        sameDiff.getVariable(FLAT_PREDICTION_TOKEN_BATCH_POSITIONS_VAR_NAME)
                .setArray(createFromArray(flatPredictionTokenBatchPositions.toArray(Integer[]::new)));

        INDArray out = sameDiff.getVariable(MODEL_OUTPUT_VAR_NAME).eval();
        double[][] softmaxResults = out.toDoubleMatrix();
        return Arrays.stream(softmaxResults)
                .map(softmaxResult -> range(0, softmaxResult.length)
                        .mapToObj(index -> Pair.of(index, softmaxResult[index]))
                        .sorted(Entry.<Integer, Double>comparingByValue().reversed())
                        .limit(limit)
                        .map(valueByIndex -> vocab.getTokenByIndex(valueByIndex.getKey()).orElse(""))
                        .toList())
                .collect(toList());
    }

    protected static class PassageTokensSequenceTagged extends
            AbstractOpenDatasetTransformerModel.PassageTokensSequence {
        protected List<String> posTags;

        public PassageTokensSequenceTagged(List<String> wordPieceTokens, List<String> posTags) {
            super(wordPieceTokens);
            this.posTags = posTags;
        }

        @Override
        public String toString() {
            return new StringJoiner(", ", PassageTokensSequenceTagged.class.getSimpleName() + "[", "]")
                    .add("posTags=" + posTags)
                    .add("\n")
                    .add("wordPieceTokens=" + wordPieceTokens)
                    .toString();
        }
    }

    @Override
    public String toString() {
        var stringJoiner = new StringJoiner("\n")
                .add("hiddenSize=" + hiddenSize)
                .add("learningRate=" + format("%.5f", learningRate))
                .add("attentionHeadsAmount=" + attentionHeadsAmount)
                .add("encoderLayersAmount=" + layersAmount)
                .add("intermediateLayerSize=" + intermediateLayerSize)
                .add("sequenceLength=" + baseSequenceLength)
                .add("batchSize=" + batchSize)
                .add("labelSmoothing=" + labelSmoothing)
                .add("percentageOfTokensToBePredicted=" + percentageOfTokensToBePredicted)
                .add("maxSizeOfWholePhrasePrediction=" + maxSizeOfWholePhrasePrediction)
                .add("percentageOfMaskingPerPrediction=" + percentageOfMaskingPerPrediction)
                .add("beta2=" + beta2)
                .add("loggingFrequency=" + loggingFrequency)
                .add("testingFrequency=" + modelTestFrequency);
        return stringJoiner.toString();
    }

    /**
     * A basic custom listener that logs the useful info (like current learning rate, accuracy during training and testing, loss)
     * and does some preventive measures (like zeroing out NaNs in weight updates, early stopping etc.)
     */
    public static class CustomListener extends BaseListener {
        private boolean earlyTrainingStop;
        private static final Condition nanCondition = Conditions.isNan();
        private final int logFrequency;
        private Instant start = now();
        private Instant loggerStart = now();

        public CustomListener(int logFrequency) {
            this.logFrequency = logFrequency;
        }

        @Override
        public void epochStart(SameDiff sd, At at) {
            super.epochStart(sd, at);
            start = now();
        }

        @Override
        public boolean isActive(Operation operation) {
            return true;
        }

        @Override
        public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
            //LOG.info("Updating {}", v.getName());

            long nanElementsAmount = update.scan(nanCondition).longValue();
            long totalElementsAmount = update.length();
            double nanRate = nanElementsAmount * 100 / (double) totalElementsAmount;
            if (nanElementsAmount > 0) {
                if ((at.iteration() + 1) % logFrequency == 0) {
                    LOG.warn("{} NaN elements ({}%) in update for '{}'. Zeroing out them all in order to avoid destroying weights",
                            nanElementsAmount, format("%.1f", nanRate), v.getName());
                }
                replaceWhere(update, 0.0, nanCondition);
            }
        }

        @Override
        public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext,
                                INDArray[] outputs) {
            if ((at.iteration() + 1) % logFrequency == 0 &&
                    op.getOutputsOfOp().stream().anyMatch(opn -> opn.contains("positionalAttentionLoss") && !opn.endsWith("grad"))) {
                LOG.info("After {} iteration POSITIONAL ATTENTION LOSS:  {}", getCounterEnding((at.iteration() + 1)),
                        format("%.1f", outputs[0].getDouble(0)));
            }

            if ((at.iteration() + 1) % logFrequency == 0 && batch != null &&
                    op.getName().startsWith("softmax") && op.getOutputsOfOp().contains(MODEL_OUTPUT_VAR_NAME)) {
                printMainLossAndAccuracy(batch.getLabels(0), outputs[0], "MAIN LOSS", at.iteration() + 1);
            }

            if ((at.iteration() + 1) % logFrequency == 0 && batch != null &&
                    op.getName().startsWith("softmax") && op.getOutputsOfOp().contains(POS_TAG_PREDICTION_LOGITS_VAR_NAME)) {
                long batchSize = batch.getFeatures(0).size(0);
                long seqLength = batch.getFeatures(0).size(1);
                var flatLabels = new NDBase().gather(batch.getLabels(2).reshape(batchSize * seqLength), batch.getLabels(1), 0);
                printMainLossAndAccuracy(flatLabels, outputs[0], "POS TAGGING LOSS", at.iteration() + 1);
            }
        }

        @Override
        public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
            //LOG.info("Done iteration in {} secs", CommonUtils.getDurationInSeconds(start));
            if ((at.iteration() + 1) % logFrequency == 0) {
                LOG.info("Done last {} epochs in {} seconds ( {} minutes )", logFrequency,
                        getDurationInSeconds(loggerStart),
                        format("%.1f", getDurationInSeconds(loggerStart) / (float) 60));
                loggerStart = now();
                System.out.println();
            }

            return earlyTrainingStop ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
        }

        private void printAverageTokenAttentionWeightsToAllTokens(MultiDataSet batch, SameDiff sd, INDArray attentionWeights) {
            String type = batch == null ? "Testing" : "Training";
            INDArray tokenIndices = batch != null ? batch.getFeatures()[0] : sd.getVariable(INPUT_VAR_NAME).getArr();

            Map<Integer, Double> mostImportantPositionsByOccurrence = getMostImportantPositionsByOccurrence(attentionWeights,
                    tokenIndices, 200, false, false);

            LOG.info("Most attended positions for NON-MASKED tokens during {} -----> {}", type,
                    mostImportantPositionsByOccurrence.entrySet().stream()
                            .sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue()))
                            .limit(5)
                            .map(entry -> format("%5s : %6.1f", format("[%d]", entry.getKey()), entry.getValue()))
                            .collect(joining(" ")));
        }

        private void printMaskedTokenAttentionWeightsToAllTokens(MultiDataSet batch, SameDiff sd, INDArray attentionWeights) {
            String type = batch == null ? "Testing" : "Training";
            INDArray tokenIndices = batch != null ? batch.getFeatures()[0] : sd.getVariable(INPUT_VAR_NAME).getArr();
            Map<Integer, Double> mostImportantPositionsByOccurrence = getMostImportantPositionsByOccurrence(attentionWeights,
                    tokenIndices, 200, false, true);

            LOG.info("Most attended positions for MASKED tokens during {} ---> {}", type,
                    mostImportantPositionsByOccurrence.entrySet().stream()
                            .sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue()))
                            .limit(5)
                            .map(entry -> format("%5s : %2.1f", format("[%d]", entry.getKey()), entry.getValue()))
                            .collect(joining("  ")));
        }

        private void printMainLossAndAccuracy(INDArray labels, INDArray predictions, String lossType, int iteration) {
            List<Double> losses = new LinkedList<>();

            for (int i = 0; i < predictions.slices(); i++) {
                losses.add(log(predictions.slice(i).getDouble(labels.getInt(i))));
            }
            double latestLoss = losses.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            if (Double.isNaN(latestLoss)) {
                earlyTrainingStop = true;
            }

            int[] actualPredictions = predictions.argMax(1).toIntVector();
            List<Boolean> predictionMatchStates = new LinkedList<>();
            for (int i = 0; i < labels.slices(); i++) {
                predictionMatchStates.add(actualPredictions[i] == labels.getInt(i));
            }
            long truePredictions = predictionMatchStates.stream().filter(aBoolean -> aBoolean).count();
            double latestAccuracy = truePredictions / (double) predictionMatchStates.size() * 100;

            LOG.info("After {} iteration for {}:  Accuracy  {}, loss  {}", getCounterEnding(iteration), lossType,
                    format("%.1f", latestAccuracy), abs(latestLoss));
        }

        @Nonnull
        private Map<Integer, Double> getMostImportantPositionsByOccurrence(INDArray attentionWeights, INDArray tokenIndices,
                                                                           int maxPosition, boolean includeCls,
                                                                           boolean onlyMaskedTokens) {
            Map<Integer, Double> mostImportantPositionsByOccurrence = new HashMap<>();

            int[] allTokenIndices = tokenIndices.getRow(0, true).toIntVector();
            INDArray attentionWeightsForSequence = attentionWeights.slice(0);
            var tokenStream = range(includeCls ? 0 : 1, allTokenIndices.length)
                    .filter(tokenPosition -> tokenPosition < maxPosition)
                    .filter(tokenPosition -> allTokenIndices[tokenPosition] != 0);
            if (onlyMaskedTokens) {
                tokenStream = tokenStream.filter(tokenPosition -> allTokenIndices[tokenPosition] == 103);
            } else {
                tokenStream = tokenStream.filter(tokenPosition -> allTokenIndices[tokenPosition] != 103);
            }
            tokenStream.forEach(tokenPosition -> {
                // Attention for each head
                for (int j = 0; j < attentionWeightsForSequence.slices(); j++) {
                    // Attention for each word of j-th head
                    double[] attentionForPredictionToken = attentionWeightsForSequence.slice(j)
                            .slice(tokenPosition)
                            .toDoubleVector();
                    range(0, attentionForPredictionToken.length)
                            .mapToObj(index -> Pair.of(index, attentionForPredictionToken[index]))
                            .sorted(Entry.<Integer, Double>comparingByValue().reversed())
                            .limit(5)
                            .forEach(valueByIndex -> {
                                int relativePosition = -(tokenPosition - valueByIndex.getKey());
                                mostImportantPositionsByOccurrence.computeIfPresent(relativePosition,
                                        (p, rating) -> valueByIndex.getValue() + rating);
                                mostImportantPositionsByOccurrence.putIfAbsent(relativePosition,
                                        valueByIndex.getValue());
                            });
                }
            });
            return mostImportantPositionsByOccurrence;
        }
    }

    public static final class Builder extends AbstractTransformerSameDiffModel.Builder<Builder, MlmTransformerSdModel> {
        private int percentageOfTokensToBePredicted = 15;
        private int percentageOfMaskingPerPrediction = 30;
        private int maxSizeOfWholePhrasePrediction = 3;
        private int minimumSequenceUtilizationPercentage = 50;
        private final List<String> englishConjunctors;
        private final PosTagsVocab posTagsVocab;
        private int miniEpochs = 2;
        private EncoderType encoderType = CLASSIC;

        public Builder(WordPieceVocab vocab, List<String> englishConjunctors, PosTagsVocab posTagsVocab) {
            super(vocab);
            this.posTagsVocab = posTagsVocab;
            this.englishConjunctors = requireNonNull(englishConjunctors).stream()
                    .map(String::toLowerCase)
                    .collect(toImmutableList());
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public Builder withEncoderType(EncoderType encoderType) {
            this.encoderType = encoderType;
            return this;
        }

        public Builder withMiniEpochs(int miniEpochs) {
            this.miniEpochs = miniEpochs;
            return this;
        }

        public Builder withPercentageOfTokensToBePredicted(int percentageOfTokensToBePredicted) {
            this.percentageOfTokensToBePredicted = percentageOfTokensToBePredicted;
            return this;
        }

        public Builder withMaxSizeOfWholePhrasePrediction(int maxSizeOfWholePhrasePrediction) {
            this.maxSizeOfWholePhrasePrediction = maxSizeOfWholePhrasePrediction;
            return this;
        }

        public Builder withPercentageOfMaskingPerPrediction(int percentageOfMaskingPerPrediction) {
            this.percentageOfMaskingPerPrediction = percentageOfMaskingPerPrediction;
            return this;
        }

        public Builder withMinimumSequenceUtilizationPercentage(int minimumSequenceUtilizationPercentage) {
            checkArgument(minimumSequenceUtilizationPercentage > 0 && minimumSequenceUtilizationPercentage <= 100,
                    "Minimum sequence utilization percentage value should be between 1 and 100");
            this.minimumSequenceUtilizationPercentage = minimumSequenceUtilizationPercentage;
            return this;
        }

        public MlmTransformerSdModel build() {
            requireNonNull(this.vocab, "TransformerEncoderModel must have a vocab set before initialization");
            try {
                var tokenizerFactory =
                        new BertWordPieceTokenizerFactory(vocab.getTokenReader(), true, true, WordPieceVocab.DEFAULT_CHAR_SET);
                return new MlmTransformerSdModel(beta2, hiddenSize, englishConjunctors, intermediateLayerSize, learningRate, labelSmoothing,
                        attentionHeadsAmount, layersAmount, percentageOfTokensToBePredicted, maxSizeOfWholePhrasePrediction,
                        percentageOfMaskingPerPrediction, sequenceLength, batchSize, miniEpochs, saveFrequencyInSteps, vocab,
                        loggingFrequency, modelTestFrequency, tokenizerFactory, minimumSequenceUtilizationPercentage, posTagsVocab,
                        encoderType);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}