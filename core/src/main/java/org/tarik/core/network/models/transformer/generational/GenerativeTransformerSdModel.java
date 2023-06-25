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

package org.tarik.core.network.models.transformer.generational;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.layers.sd.transformer.TransformerDecoderGraphGenerator;
import org.tarik.core.network.layers.sd.transformer.attention.AttentionLayerGraphGenerator;
import org.tarik.core.network.models.transformer.AbstractOpenDatasetTransformerModel;
import org.tarik.core.network.models.transformer.AbstractOpenDatasetTransformerModel.PassageTokensSequence;
import org.tarik.core.network.models.transformer.AbstractTransformerSameDiffModel;
import org.tarik.core.vocab.WordPieceVocab;
import org.tarik.utils.visualisation.charts.ConvergenceChartPlotter;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.Serial;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.*;
import static java.lang.String.format;
import static java.time.Instant.now;
import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.ofNullable;
import static java.util.concurrent.Executors.newSingleThreadExecutor;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;
import static org.nd4j.linalg.api.buffer.DataType.*;
import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.indexing.BooleanIndexing.replaceWhere;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.tarik.utils.CommonUtils.getCounterEnding;
import static org.tarik.utils.CommonUtils.getDurationInSeconds;
import static org.tarik.utils.SdUtils.gather;
import static org.tarik.utils.SdUtils.softmaxForLossCalculation;
import static org.tarik.utils.visualisation.charts.ConvergenceChartPlotter.newChartPlotter;

/**
 * <p>
 * Concrete implementation of {@link AbstractOpenDatasetTransformerModel} aimed at performing the token generation tasks. This is a
 * GPT-like model which is based on different papers and articles.
 * </p>
 * <p>
 * The model is created using the builder pattern. In order to start the model's training,
 * {@link GenerativeTransformerSdModel#train(BiConsumer, BiConsumer, int, Consumer, Supplier)} method must be called.
 * In order to evaluate the model's accuracy, {@link GenerativeTransformerSdModel#test(Map)} method must be called. The resulting accuracy
 * will be logged.
 * In order to trigger the execute inference, {@link GenerativeTransformerSdModel#predictNextToken(String, int)} method must be called.
 * </p>
 * <p>
 * The model uses a custom {@link GenerativeTransformerSdModel.CustomListener} which allows to log different debug and execution data in order to show the
 * current training progress. It's also responsible for plotting the accuracy results.
 * </p>
 */

public class GenerativeTransformerSdModel extends AbstractOpenDatasetTransformerModel<GenerativeTransformerSdModel, PassageTokensSequence> {

    @Serial
    private static final long serialVersionUID = 3463116983031854533L;
    private static final Logger LOG = LoggerFactory.getLogger(GenerativeTransformerSdModel.class);
    protected static final String MODEL_FILE_NAME = "GenerativeTransformerModel";
    protected static final String SELF_ATTENTION_CAUSAL_MASKS_VAR_NAME = "selfAttentionCausalMasks";
    private static final String DECODER_INPUT_VAR_NAME = "decoderInputTokenVocabIndices";
    protected static final String DECODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME = "decoderPositionalEmbeddingsMatrix";
    public static final String MODEL_OUTPUT_VAR_NAME = "finalLayerResult";
    public static final String FLAT_TARGET_BATCH_POSITIONS_VAR_NAME = "FLAT_TARGET_BATCH_POSITIONS";
    public static final String FLAT_TARGET_TOKEN_VOCAB_INDICES_VAR_NAME = "FLAT_TARGET_TOKEN_VOCAB_INDICES";
    public static final String DECODER_ID = "decoder";
    private static final Random random = new Random();
    private final int percentageOfTokensToBePredicted;
    private final HashSet<TokenUsageStats> collectedTokenUsageStats = new HashSet<>();
    private transient CustomListener defaultListener;

    @Override
    protected PassageTokensSequence generateTruncatedPassageTokensSequence(LinkedList<String> sentencesQueue) {
        List<String> wordPieceTokens = new LinkedList<>();
        while (!sentencesQueue.isEmpty() && wordPieceTokens.size() < totalSequenceCapacity) {
            var sentence = sentencesQueue.pollFirst();
            wordPieceTokens.addAll(tokenizerFactory.create(sentence).getTokens());
        }

        return new PassageTokensSequence(wordPieceTokens);
    }

    private GenerativeTransformerSdModel(int sequenceLength, int encoderBatchSize, double beta2, List<String> englishConjunctors,
                                         int hiddenSize, int modelTestFrequency, WordPieceVocab vocab, int loggingFrequency,
                                         int modelSaveFrequency, double learningRate, int layersAmount, int attentionHeadsAmount,
                                         int intermediateLayerSize, float dropoutRate, ISchedule optimizerSchedule,
                                         double labelSmoothing, boolean fixWeights, boolean fixTokenEmbeddings,
                                         BertWordPieceTokenizerFactory tokenizerFactory, int minimumSequenceUtilizationPercentage,
                                         int percentageOfTokensToBePredicted) {
        super(encoderBatchSize, sequenceLength, modelTestFrequency, loggingFrequency, learningRate, hiddenSize, layersAmount,
                attentionHeadsAmount, intermediateLayerSize, labelSmoothing, fixWeights, fixTokenEmbeddings, modelSaveFrequency,
                beta2, dropoutRate, englishConjunctors, minimumSequenceUtilizationPercentage, sequenceLength - 1);
        this.vocab = requireNonNull(vocab);
        this.percentageOfTokensToBePredicted = percentageOfTokensToBePredicted;
        this.optimizerSchedule = optimizerSchedule;
        this.tokenizerFactory = tokenizerFactory;

        buildCompleteSdGraph();
        LOG.info(""" 
                Created a Generative Transformer NLP Model with the following params:
                -----------------------------------------------------------------------------
                {}
                -----------------------------------------------------------------------------
                """, this);
    }

    @Override
    protected TrainingConfig getTrainingConfig() {
        String[] labels = new String[]{FLAT_TARGET_TOKEN_VOCAB_INDICES_VAR_NAME, FLAT_TARGET_BATCH_POSITIONS_VAR_NAME};
        return new TrainingConfig.Builder()
                .updater(getUpdater())
                .dataSetFeatureMapping(DECODER_INPUT_VAR_NAME)
                .dataSetLabelMapping(labels)
                .dataSetFeatureMaskMapping(SELF_ATTENTION_CAUSAL_MASKS_VAR_NAME)
                .build();
    }

    public int getSequenceLengthWithoutSeparators() {
        return baseSequenceLength;
    }

    private MultiDataSet getMultiDataSet(INDArray batchDecoderTokenVocabIndices, List<Integer> flatTargetBatchPositions,
                                         List<Integer> flatTargetTokenVocabIndices, INDArray decoderBatchSelfAttentionCausalMasks) {
        var labels = new INDArray[]{createFromArray(flatTargetTokenVocabIndices.toArray(Integer[]::new)),
                createFromArray(flatTargetBatchPositions.toArray(Integer[]::new))};
        var features = new INDArray[]{batchDecoderTokenVocabIndices};
        return new org.nd4j.linalg.dataset.MultiDataSet(features, labels, new INDArray[]{decoderBatchSelfAttentionCausalMasks},
                new INDArray[]{Nd4j.empty(), Nd4j.empty()});
    }

    @Override
    protected String getModelFileName() {
        return MODEL_FILE_NAME;
    }

    @Override
    protected GenerativeTransformerSdModel getInstance() {
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
                Building a Generative Transformer NLP SameDiff model with the following params:
                 - number of decoder blocks: {}
                 - max sequence length: {}
                 - token embeddings and hidden layer size: {}
                 - vocab size: {}
                 - attention heads amount: {}
                """, layersAmount, baseSequenceLength, hiddenSize, vocab.size(), attentionHeadsAmount);

        // Token embeddings
        var tokenEmbeddingsMatrix = sd.var("tokenEmbeddingsMatrix",
                new XavierInitScheme('c', vocab.size(), hiddenSize), FLOAT, vocab.size(), hiddenSize);
        var selfAttentionCausalMasks = sd.placeHolder(SELF_ATTENTION_CAUSAL_MASKS_VAR_NAME,
                FLOAT, -1, 1, baseSequenceLength, baseSequenceLength);
        var decoderInputTokenVocabIndices = sd.placeHolder(DECODER_INPUT_VAR_NAME, INT32, -1, baseSequenceLength);
        var decoderPositionalEmbeddingsMatrix = sd.var(DECODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME,
                initializePositionalEmbeddings(baseSequenceLength)).convertToConstant();

        if (fixTokenEmbeddings) {
            tokenEmbeddingsMatrix.convertToConstant();
            decoderPositionalEmbeddingsMatrix.convertToConstant();
        }

        var batchDecoderInputTokenEmbeddings =
                gather(sd, "batchDecoderInputTokenEmbeddings", tokenEmbeddingsMatrix, decoderInputTokenVocabIndices, 0);

        // Building decoder's graph
        var transformerDecoderGraphGenerator = getTransformerDecoderGraphGenerator();
        var decoderOutput = transformerDecoderGraphGenerator.generateGraph(sd, batchDecoderInputTokenEmbeddings,
                selfAttentionCausalMasks, decoderPositionalEmbeddingsMatrix);

        // flatPredictionTokenBatchPositions is a vector which contains the batch position indices of
        // the tokens which need to be predicted for the whole batch
        var flatTargetBatchPositions = sd.placeHolder(FLAT_TARGET_BATCH_POSITIONS_VAR_NAME, INT32, -1);

        // Collecting final word embeddings for the positions which need to be predicted
        var predictionTokenEmbeddings = gather(sd, "predictionTokenEmbeddings", decoderOutput, flatTargetBatchPositions, 0);

        //Output layer and loss
        var finalLayerResult = sd.mmul(MODEL_OUTPUT_VAR_NAME, predictionTokenEmbeddings, tokenEmbeddingsMatrix, false, true, false);
        var flatTargetTokenVocabIndices = sd.placeHolder(FLAT_TARGET_TOKEN_VOCAB_INDICES_VAR_NAME, INT32, -1);
        var oneHotLabels = sd.oneHot("oneHotLabels", flatTargetTokenVocabIndices, vocab.size());
        var oneHotSmoothLabels = oneHotLabels.mul(1 - labelSmoothing).plus(labelSmoothing / vocab.size())
                .rename("oneHotSmoothLabels");
        var logitPredictions = softmaxForLossCalculation(sd, "modelPredictionLogits", finalLayerResult, 1, oneHotSmoothLabels);
        sd.math.log(logitPredictions).mul(oneHotLabels).sum(1).mean(0).neg("loss").markAsLoss();

        this.sameDiff = sd;
    }

    @Nonnull
    private TransformerDecoderGraphGenerator getTransformerDecoderGraphGenerator() {
        return new TransformerDecoderGraphGenerator.Builder(DECODER_ID)
                .withAttentionHeadsAmount(this.attentionHeadsAmount)
                .withLayersAmount(this.layersAmount)
                .withHiddenSize(hiddenSize)
                .withIntermediateLayerSize(this.intermediateLayerSize)
                .withSequenceLength(this.baseSequenceLength)
                .withFixWeights(fixWeights)
                .withDropoutRate(dropout)
                .build();
    }

    @Override
    public Optional<GenerativeTransformerSdModel> loadModelDataFromFile(Path zipFilePath, boolean loadUpdater)
            throws IOException {
        var loadedModel = super.loadModelDataFromFile(zipFilePath, loadUpdater);
        loadedModel.map(model -> model.collectedTokenUsageStats).ifPresent(stats -> {
            this.collectedTokenUsageStats.clear();
            this.collectedTokenUsageStats.addAll(stats);
        });
        return loadedModel;
    }

    public void train(BiConsumer<GenerativeTransformerSdModel, Long> incompleteEpochModelSaver,
                      BiConsumer<GenerativeTransformerSdModel, Long> completeEpochModelSaver,
                      int totalStepsAmount, Consumer<GenerativeTransformerSdModel> evaluator,
                      Supplier<Boolean> dataLimitReached)
            throws IOException, CloneNotSupportedException, ExecutionException, InterruptedException {
        requireNonNull(dataProvider, "Model can't be trained without a data provider");
        int processedStepsAmount = sameDiff.getTrainingConfig().getIterationCount();
        ExecutorService modelSaveExecutor = newSingleThreadExecutor();

        String plotterName = format("%d_%s", batchSize, "generative-transformer");
        // Listeners
        defaultListener = new CustomListener(loggingFrequency, plotterName, vocab, baseSequenceLength);
        sameDiff.setListeners(defaultListener);

        LinkedList<PassageTokensSequence> remainingTokenSentences = new LinkedList<>();

        while (processedStepsAmount <= totalStepsAmount) {
            if ((processedStepsAmount + 1) % 10 == 0) {
                LOG.info("Starting {} training step having a buffer of {} passages from previous one",
                        getCounterEnding(processedStepsAmount + 1), remainingTokenSentences.size());
            }

            var batchedTokenSentences = getNewTokenSentencesBatch(remainingTokenSentences,
                    fetchedBlocks -> fetchedBlocks.size() > batchSize || dataLimitReached.get());
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
            var actualBatchSize = min(batchedTokenSentences.size(), batchSize);
            INDArray batchTokenVocabIndices = create(INT32, actualBatchSize, baseSequenceLength);
            //INDArray batchSelfAttentionCausalMasks = create(INT8, actualBatchSize, baseSequenceLength);
            INDArray batchSelfAttentionCausalMasks = create(INT8, actualBatchSize, 1, baseSequenceLength, baseSequenceLength);
            List<Integer> flatTargetBatchPositions = new LinkedList<>();
            List<Integer> flatTargetVocabIndices = new LinkedList<>();

            range(0, actualBatchSize).forEach(batchPosition -> {
                addSequenceData(batchPosition, batchedTokenSentences.get(batchPosition), flatTargetVocabIndices,
                        batchTokenVocabIndices, batchSelfAttentionCausalMasks, collectedTokenUsageStats, flatTargetBatchPositions);
            });

            var multiDataSet = getMultiDataSet(batchTokenVocabIndices, flatTargetBatchPositions,
                    flatTargetVocabIndices, batchSelfAttentionCausalMasks);
            sameDiff.fit(new SingletonMultiDataSetIterator(multiDataSet), 1);
            ++processedStepsAmount;

            System.gc();

            if (defaultListener.earlyTrainingStop) {
                LOG.warn("Interrupting the training at {} step because early stop was required",
                        getCounterEnding(processedStepsAmount));
                defaultListener.earlyTrainingStop = false;
                break;
            }

            if (processedStepsAmount % modelTestFrequency == 0) {
                // Because inference with sufficient data takes some time, saving the model in parallel is a good
                // option of saving time, unless it's the last epoch or near the intermittent save schedule - then
                // the model will be saved anyway
                if (modelTestFrequency % saveFrequencyInSteps == 0) {
                    // Testing and saving model concurrently. As testing is a read-only operation, should be fine.
                    int stepsDone = processedStepsAmount;
                    var saveFuture = modelSaveExecutor.submit(() -> saveModelDuringTraining(incompleteEpochModelSaver, stepsDone));
                    evaluator.accept(this);
                    saveFuture.get();
                } else {
                    evaluator.accept(this);
                }

                sameDiff.getSessions().clear();
                System.gc();
            }

            if (processedStepsAmount % saveFrequencyInSteps == 0 && modelTestFrequency != saveFrequencyInSteps) {
                LOG.info("Saving a model after {} step", getCounterEnding(processedStepsAmount));
                completeEpochModelSaver.accept(this, categoriesBufferFromCurrentMiniEpoch);
            }
        }
        LOG.info("Saving a model at the end of training");
        completeEpochModelSaver.accept(this, 0L);
        LOG.info("Completed training after {} steps", processedStepsAmount);

    }

    public void test(Map<String[], String> sentenceTokensByPredictionToken) {
        Instant start = now();
        List<Double> allAccuracies = new LinkedList<>();
        var remainingWordsBySentenceQueue = new LinkedList<>(sentenceTokensByPredictionToken.entrySet());

        while (!remainingWordsBySentenceQueue.isEmpty()) {
            List<Pair<String[], String>> batchWordsBySequence = new LinkedList<>();
            while (batchWordsBySequence.size() < batchSize && !remainingWordsBySentenceQueue.isEmpty()) {
                var maskedWordBySequence = remainingWordsBySentenceQueue.poll();
                if (maskedWordBySequence.getKey().length > 0) {
                    batchWordsBySequence.add(Pair.of(maskedWordBySequence.getKey(), maskedWordBySequence.getValue()));
                }
            }

            List<Double> accuracies = new LinkedList<>();
            var out = getNextTokenPredictionLogits(batchWordsBySequence.stream().map(pair -> Arrays.asList(pair.getKey())).toList());
            var results = getNextTokenPredictionResults(out,
                    batchWordsBySequence.stream().map(Pair::getValue).toList(), 3);
            List<Integer> winners = random.ints(0, results.size()).distinct().limit(min(3, results.size())).boxed().toList();
            for (int i = 0; i < results.size(); i++) {
                var predictionResult = results.get(i);
                accuracies.add(predictionResult.accuracy());
                if (winners.contains(i)) {
                    LOG.info("Expected Word: {},  Top predictions: {}", predictionResult.label(),
                            predictionResult.topPredictedProbabilitiesByTokens().stream()
                                    .map(e -> format("<<%s>> -> %.4f ", e.getKey(), e.getValue())).collect(joining(", ")));
                    LOG.info("Full passage: " + String.join(" ", batchWordsBySequence.get(i).getKey()));
                    LOG.info("Accuracy: " + predictionResult.accuracy());
                    System.out.println();
                }
            }

            allAccuracies.addAll(accuracies);
        }
        var accuracy = allAccuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        ofNullable(defaultListener).ifPresent(listener -> listener.addTestAccuracyPlotterData(
                this.sameDiff.getTrainingConfig().getIterationCount(), accuracy));
        LOG.info("TOTAL TEST accuracy: {}", format("%.1f", accuracy));
        LOG.info("Done inference in  {} seconds", getDurationInSeconds(start));
    }

    public List<String> predictNextToken(String sentence, int candidatesLimit) {
        this.sameDiff.setListeners(new CustomListener(loggingFrequency, null, vocab, baseSequenceLength));
        var sentenceTokens = new LinkedList<>(tokenizerFactory.create(sentence.trim()).getTokens());
        INDArray out = getNextTokenPredictionLogits(List.of(sentenceTokens));
        return getSingleSequenceLastTokenPredictions(out, candidatesLimit);
    }

    private void addSequenceData(int batchPosition, PassageTokensSequence passageTokensSequence,
                                 List<Integer> flatPredictionTokenVocabIndices,
                                 INDArray tokenVocabIndicesCollector, INDArray selfAttentionCausalMasksCollector,
                                 Set<TokenUsageStats> collectedTokenUsageStats,
                                 List<Integer> flatPredictionTokenBatchPositions) {
        var tokens = passageTokensSequence.getWordPieceTokens();
        int expectedPredictionsAmount = max(1, (tokens.size() * percentageOfTokensToBePredicted) / 100);
        List<Integer> predictionPositions = getPredictionPositions(tokens, collectedTokenUsageStats, expectedPredictionsAmount, 1);

        addLabels(batchPosition, flatPredictionTokenVocabIndices, flatPredictionTokenBatchPositions, tokens, predictionPositions);

        addSequenceFeaturesAndMasks(batchPosition, tokens, tokenVocabIndicesCollector, selfAttentionCausalMasksCollector);
    }

    private void addLabels(int batchPosition, List<Integer> flatPredictionTokenVocabIndices,
                           List<Integer> flatPredictionTokenBatchPositions, List<String> tokens, List<Integer> predictionTokenPositions) {
        predictionTokenPositions.forEach(predictionTokenPosition -> {
            // We take the representation (encoding) of the previous token in order to predict the next one
            checkArgument(predictionTokenPosition > 0, "Prediction position must be > 0");
            if (predictionTokenPosition < tokens.size()) {
                flatPredictionTokenVocabIndices.add(vocab.getTokenIndex(tokens.get(predictionTokenPosition)).orElseThrow());
            } else {
                flatPredictionTokenVocabIndices.add(vocab.getTokenIndex(tokens.get(tokens.size() - 1)).orElseThrow());
            }
            flatPredictionTokenBatchPositions.add(baseSequenceLength * batchPosition + predictionTokenPosition - 1);
        });
    }

    private void addLabelsForInference(int batchPosition, List<Integer> flatPredictionTokenBatchPositions,
                                       List<Integer> predictionTokenPositions) {
        predictionTokenPositions.forEach(predictionTokenPosition -> {
            // We take the representation (encoding) of the previous token in order to predict the next one
            checkArgument(predictionTokenPosition > 0, "Prediction position must be > 0");
            flatPredictionTokenBatchPositions.add(baseSequenceLength * batchPosition + predictionTokenPosition - 1);
        });
    }

    private List<Integer> getPredictionPositions(List<String> sequence, Set<TokenUsageStats> collectedTokenUsageStats,
                                                 int expectedPredictionsAmount, int amountOfFirstPositionsToIgnore) {
        int distanceBetweenPredictions = sequence.size() / expectedPredictionsAmount;
        checkArgument(amountOfFirstPositionsToIgnore > 0,
                "Generative models can't predict the first token it's the one to provide the first hidden state");

        Map<Integer, String> candidateTokensByPosition = range(0, min(sequence.size(), getSequenceLengthWithoutSeparators()))
                .skip(amountOfFirstPositionsToIgnore)
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

        if (predictionPositions.size() < expectedPredictionsAmount) {
            // If there's still some place for predictions - choosing the tokens which have the lowest prediction usage
            addLeastFrequentlyPredictedTokenPositions(expectedPredictionsAmount, collectedTokenUsageStats, distanceBetweenPredictions,
                    candidateTokensByPosition, notYetKnownTokensByPosition, predictionPositions);
        }

        return predictionPositions.stream().sorted().toList();
    }

    private void addSequenceFeaturesAndMasks(int batchIndex, List<String> tokens, INDArray decoderBatchTokenVocabIndices,
                                             INDArray decoderBatchFeatureMasks) {
        INDArray sequenceTokenVocabIndices = create(INT32, baseSequenceLength);

        range(0, min(tokens.size(), baseSequenceLength))
                .forEach(index -> sequenceTokenVocabIndices.putScalar(index, vocab.getTokenIndex(tokens.get(index)).orElseThrow()));
        for (int i = tokens.size(); i < baseSequenceLength; i++) {
            sequenceTokenVocabIndices.putScalar(i, vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow());
        }

        var tokenAttentionMasks = getTokenCausalSelfAttentionMasks(tokens.size() - 1);
        decoderBatchFeatureMasks.put(new INDArrayIndex[]{point(batchIndex)}, tokenAttentionMasks);
        decoderBatchTokenVocabIndices.putRow(batchIndex, sequenceTokenVocabIndices);
    }

    private List<PredictionResult> getNextTokenPredictionResults(INDArray out, List<String> labelWords, int topPredictionsDepth) {
        double[][] predictionLogitsMatrix = getPredictionLogitsMatrix(out);
        List<PredictionResult> collector = new LinkedList<>();
        for (int i = 0; i < predictionLogitsMatrix.length; i++) {
            List<Pair<String, Double>> probabilitiesByToken =
                    getTopPredictionProbabilitiesByToken(topPredictionsDepth, predictionLogitsMatrix[i]);
            Map<String, Integer> predictionsByPosition = range(0, probabilitiesByToken.size())
                    .boxed()
                    .collect(toMap(ind -> probabilitiesByToken.get(ind).getKey(), identity()));

            var labelWord = labelWords.get(i);
            double accuracy = ofNullable(predictionsByPosition.get(labelWord))
                    .map(predictionPosition -> 100 / ((double) predictionPosition + 1)).orElse(0d);
            collector.add(new PredictionResult(probabilitiesByToken, accuracy, labelWord));
        }
        return collector;
    }

    private List<String> getSingleSequenceLastTokenPredictions(INDArray out, int topPredictionsDepth) {
        double[] predictionLogitsMatrix = getPredictionLogitsMatrix(out)[0];
        return getTopPredictionProbabilitiesByToken(topPredictionsDepth, predictionLogitsMatrix).stream()
                .map(Pair::getKey)
                .toList();
    }

    private double[][] getPredictionLogitsMatrix(INDArray out) {
        var predictionLogits = Nd4j.nn().softmax(out, 1);
        return predictionLogits.toDoubleMatrix();
    }

    @Nonnull
    private List<Pair<String, Double>> getTopPredictionProbabilitiesByToken(int topPredictionsDepth, double[] predictionLogitsMatrix) {
        return range(0, predictionLogitsMatrix.length)
                .mapToObj(index -> Pair.of(index, predictionLogitsMatrix[index]))
                .sorted(Entry.<Integer, Double>comparingByValue().reversed())
                .limit(topPredictionsDepth)
                .map(probabilityByPosition -> Pair.of(vocab.getTokenByIndex(probabilityByPosition.getKey()).orElse(""),
                        probabilityByPosition.getValue()))
                .toList();
    }

    private INDArray getNextTokenPredictionLogits(List<List<String>> batchedSequenceTokens) {
        var batchSize = batchedSequenceTokens.size();
        INDArray decoderBatchTokenVocabIndices = create(INT32, batchSize, baseSequenceLength);
        INDArray batchSelfAttentionCausalMasks = create(INT8, batchSize, 1, baseSequenceLength, baseSequenceLength);
        List<Integer> flatTargetBatchPositions = new LinkedList<>();

        for (int i = 0; i < batchedSequenceTokens.size(); i++) {
            var tokens = batchedSequenceTokens.get(i);
            var predictionPositions = List.of(tokens.size());
            addLabelsForInference(i, flatTargetBatchPositions, predictionPositions);

            addSequenceFeaturesAndMasks(i, tokens, decoderBatchTokenVocabIndices, batchSelfAttentionCausalMasks);
        }

        sameDiff.getVariable(DECODER_INPUT_VAR_NAME).setArray(decoderBatchTokenVocabIndices);
        sameDiff.getVariable(SELF_ATTENTION_CAUSAL_MASKS_VAR_NAME).setArray(batchSelfAttentionCausalMasks);
        var batchPositions = createFromArray(flatTargetBatchPositions.toArray(Integer[]::new));
        sameDiff.getVariable(FLAT_TARGET_BATCH_POSITIONS_VAR_NAME).setArray(batchPositions);

        return sameDiff.getVariable(MODEL_OUTPUT_VAR_NAME).eval();
    }

    @Override
    public String toString() {
        return new StringJoiner("\n")
                .add("batchSize=" + batchSize)
                .add("sequenceLength=" + baseSequenceLength)
                .add("modelTestFrequency=" + modelTestFrequency)
                .add("loggingFrequency=" + loggingFrequency)
                .add("learningRate=" + learningRate)
                .add("hiddenSize=" + hiddenSize)
                .add("encoderLayersAmount=" + layersAmount)
                .add("attentionHeadsAmount=" + attentionHeadsAmount)
                .add("intermediateLayerSize=" + intermediateLayerSize)
                .add("labelSmoothing=" + labelSmoothing)
                .add("fixEncoderWeights=" + fixWeights)
                .add("fixTokenEmbeddings=" + fixTokenEmbeddings)
                .add("saveFrequencyInSteps=" + saveFrequencyInSteps)
                .add("beta2=" + beta2)
                .add("dropout=" + dropout)
                .add("minimumSequenceUtilizationPercentage=" + minimumSequenceUtilizationPercentage)
                .add("totalSequenceCapacity=" + totalSequenceCapacity)
                .toString();
    }

    private record PredictionResult(List<Pair<String, Double>> topPredictedProbabilitiesByTokens, double accuracy, String label) {
    }

    /**
     * A basic custom listener that logs the useful info (like current learning rate, accuracy during training and testing, loss)
     * and does some preventive measures (like zeroing out NaNs in weight updates, early stopping etc.)
     */
    public static class CustomListener extends BaseListener {
        private boolean earlyTrainingStop;
        private static final Condition nanCondition = Conditions.isNan();
        private final int logFrequency;
        private final List<Double> latestTrainAccuracies = new LinkedList<>();
        private final List<Double> latestLosses = new LinkedList<>();
        private Instant start = now();
        private final ConvergenceChartPlotter plotter;
        private static final String TRAIN_ACCURACY = "TRAIN Accuracy";
        private static final String LOSS = "Loss";
        private static final String TEST_ACCURACY = "TEST Accuracy";
        private final WordPieceVocab vocab;
        private final int sequenceLength;

        public CustomListener(int logFrequency, String plotterName, WordPieceVocab vocab, int sequenceLength) {
            this.logFrequency = logFrequency;
            plotter = plotterName == null ? null : newChartPlotter(plotterName, List.of(LOSS, TRAIN_ACCURACY, TEST_ACCURACY));
            this.vocab = vocab;
            this.sequenceLength = sequenceLength;
        }

        public void addTestAccuracyPlotterData(int iteration, double accuracy) {
            plotter.addData(TEST_ACCURACY, iteration, accuracy);
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
            if ((timeToLog(at.iteration() + 1) || batch == null) &&
                    op.getOutputsOfOp().stream()
                            .anyMatch(v -> v.contains(AttentionLayerGraphGenerator.ATTENTION_SCORES_VAR_NAME) && !v.endsWith("grad"))) {
                printTokenAttentionWeightsToAllTokens(batch, sd, outputs[0]);
            }

            if (batch != null && op.getOutputsOfOp().contains("modelPredictionLogits")) {
                Pair<Double, Double> accuracyByLoss = getAccuracyByLoss(batch, outputs[0]);
                latestLosses.add(accuracyByLoss.getKey());
                latestTrainAccuracies.add(accuracyByLoss.getValue());
                if (timeToLog(at.iteration() + 1)) {
                    double averageLoss = abs(latestLosses.stream().mapToDouble(Double::doubleValue).average().orElse(0));
                    double averageAccuracy = latestTrainAccuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                    LOG.info("After {} iteration:   Loss {},    Accuracy {}",
                            getCounterEnding(at.iteration() + 1), format("%.5f", averageLoss), format("%.1f", averageAccuracy));
                    if (plotter != null) {
                        plotter.addData(LOSS, at.iteration() + 1, averageLoss);
                        plotter.addData(TRAIN_ACCURACY, at.iteration() + 1, averageAccuracy);
                    }

                    var batchVocabIndices = batch.getFeatures(0).toIntMatrix();
                    var labels = batch.getLabels(0).toIntVector();
                    var flatBatchPositions = batch.getLabels(1).toIntVector();
                    int[] actualPredictions = outputs[0].argMax(1).toIntVector();
                    var counter = 0;

                    for (int i = 0; i < labels.length && counter < 3; i++) {
                        if (actualPredictions[i] == labels[i]) {
                            var targetTokenPosition = flatBatchPositions[i] + 1;
                            var batchNumber = targetTokenPosition / sequenceLength;
                            var batchPosition = targetTokenPosition % sequenceLength;
                            var sequenceVocabIndices = batchVocabIndices[batchNumber];
                            var tokens = range(0, sequenceVocabIndices.length)
                                    .limit(batchPosition)
                                    .mapToObj(position -> vocab.getTokenByIndex(sequenceVocabIndices[position]).orElseThrow())
                                    .collect(Collectors.joining(" "));
                            var token = vocab.getTokenByIndex(labels[i]).orElseThrow();
                            LOG.info("\nSuccessful prediction passage: {}\nToken:  {}", tokens, token);
                            counter++;
                        }
                    }

                    latestLosses.clear();
                    latestTrainAccuracies.clear();
                }
            }
        }

        @Override
        public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
            //LOG.info("Done iteration in {} secs", getDurationInMillis(start));
            if (timeToLog(at.iteration() + 1)) {
                System.out.println();
                double learningRate = sd.getUpdaterMap().values().stream()
                        .map(GradientUpdater::getConfig)
                        .mapToDouble(updater -> updater.getLearningRate(at.iteration(), at.epoch()))
                        .average().orElse(0);
                LOG.info("Learning rate : {}", learningRate);
            }

            return earlyTrainingStop ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
        }

        private boolean timeToLog(int step) {
            return step % logFrequency == 0;
        }

        private Pair<Double, Double> getAccuracyByLoss(MultiDataSet batch, INDArray predictions) {
            INDArray labels = batch.getLabels(0);
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
            long truePredictions = predictionMatchStates.stream()
                    .filter(aBoolean -> aBoolean)
                    .count();
            double accuracy = truePredictions / (double) predictionMatchStates.size() * 100;

            return Pair.of(latestLoss, accuracy);
        }

        private void printTokenAttentionWeightsToAllTokens(MultiDataSet batch, SameDiff sd, INDArray attentionWeights) {
            String type = batch == null ? "Testing" : "Training";
            INDArray tokenIndices = batch != null ? batch.getFeatures()[0] : sd.getVariable(DECODER_INPUT_VAR_NAME).getArr();
            INDArray positionLabels = batch != null ? batch.getLabels(1) : sd.getVariable(FLAT_TARGET_BATCH_POSITIONS_VAR_NAME).getArr();
            var firstSequencePredictionPositions =
                    stream(positionLabels.toIntVector()).filter(i -> i < tokenIndices.slice(0).slices()).boxed().toList();
            Map<Integer, Double> mostImportantPositionsByOccurrence =
                    getSequenceMostAttendedPositionsByOccurrence(attentionWeights.slice(0),
                            tokenIndices.toIntMatrix()[0], 200, firstSequencePredictionPositions);

            LOG.info("Attention for predicted tokens during {} ---> {}", type,
                    mostImportantPositionsByOccurrence.entrySet().stream()
                            .sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue()))
                            .limit(5)
                            .map(entry -> format("%5s : %2.1f", format("[%d]", entry.getKey()), entry.getValue()))
                            .collect(joining("  ")));
        }


        private Map<Integer, Double> getSequenceMostAttendedPositionsByOccurrence(INDArray attentionWeightsForSequence,
                                                                                  int[] allTokenIndices, int maxPosition,
                                                                                  List<Integer> targetTokenPositions) {
            Map<Integer, Double> mostImportantPositionsByOccurrence = new HashMap<>();
            var tokenStream = range(0, allTokenIndices.length)
                    .filter(tokenPosition -> tokenPosition < maxPosition)
                    .filter(tokenPosition -> allTokenIndices[tokenPosition] != 0)
                    .filter(targetTokenPositions::contains);
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

    public static final class Builder extends AbstractTransformerSameDiffModel.Builder<Builder, GenerativeTransformerSdModel> {
        private final List<String> englishConjunctors;
        private int minimumSequenceUtilizationPercentage = 80;
        private int percentageOfTokensToBePredicted = 20;

        public Builder(WordPieceVocab vocab, List<String> englishConjunctors) {
            super(vocab);
            this.englishConjunctors = requireNonNull(englishConjunctors).stream()
                    .map(String::toLowerCase)
                    .collect(toImmutableList());
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public Builder withPercentageOfTokensToBePredicted(int percentageOfTokensToBePredicted) {
            this.percentageOfTokensToBePredicted = percentageOfTokensToBePredicted;
            return this;
        }

        public Builder withMinimumSequenceUtilizationPercentage(int minimumSequenceUtilizationPercentage) {
            this.minimumSequenceUtilizationPercentage = minimumSequenceUtilizationPercentage;
            return this;
        }

        public GenerativeTransformerSdModel build() {
            requireNonNull(this.vocab, "Model must have a vocab set before initialization");
            try {
                var tokenizerFactory =
                        new BertWordPieceTokenizerFactory(vocab.getTokenReader(), true, true, WordPieceVocab.DEFAULT_CHAR_SET);
                return new GenerativeTransformerSdModel(sequenceLength, batchSize, beta2, englishConjunctors, hiddenSize,
                        modelTestFrequency, vocab, loggingFrequency, saveFrequencyInSteps, learningRate, layersAmount, attentionHeadsAmount,
                        intermediateLayerSize, dropout, optimizerSchedule, labelSmoothing, fixWeights, fixTokenEmbeddings, tokenizerFactory,
                        minimumSequenceUtilizationPercentage, percentageOfTokensToBePredicted);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}