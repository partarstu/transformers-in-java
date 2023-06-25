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

package org.tarik.core.network.models.transformer.question_answering;

import com.google.common.collect.ImmutableSet;
import org.apache.commons.lang3.tuple.Pair;
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
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.layers.sd.transformer.TransformerEncoderGraphGenerator;
import org.tarik.core.network.layers.sd.transformer.TransformerQaDecoderGraphGenerator;
import org.tarik.core.network.models.transformer.AbstractTransformerSameDiffModel;
import org.tarik.core.network.models.transformer.question_answering.sequence.IQaTokenSequence;
import org.tarik.core.vocab.WordPieceVocab;
import org.tarik.utils.visualisation.charts.ConvergenceChartPlotter;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.Serial;
import java.time.Instant;
import java.util.*;
import java.util.Map.Entry;
import java.util.function.Consumer;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.min;
import static java.lang.Math.*;
import static java.lang.String.format;
import static java.time.Instant.now;
import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.empty;
import static java.util.Optional.ofNullable;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.toCollection;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;
import static org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor.reconstructFromTokens;
import static org.nd4j.linalg.api.buffer.DataType.*;
import static org.nd4j.linalg.factory.Nd4j.*;
import static org.nd4j.linalg.indexing.BooleanIndexing.replaceWhere;
import static org.nd4j.linalg.learning.config.Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY;
import static org.nd4j.linalg.learning.config.Adam.DEFAULT_ADAM_EPSILON;
import static org.tarik.utils.CommonUtils.getCounterEnding;
import static org.tarik.utils.CommonUtils.getDurationInMinutes;
import static org.tarik.utils.SdUtils.*;
import static org.tarik.utils.visualisation.charts.ConvergenceChartPlotter.newChartPlotter;

/**
 * <p>
 * An experimental generative model inherited from {@link AbstractTransformerSameDiffModel} aimed at performing the Question-Answering
 * tasks. This is a classical Transformers-based model which has both encoder and decoder stacks.
 * </p>
 * <p>
 * The model is created using the builder pattern. In order to start the model's training,
 * {@link GenerativeQuestionAnsweringModel#train(ArrayList, Consumer, int, Consumer)} method must be called.
 * In order to evaluate the model's accuracy, {@link GenerativeQuestionAnsweringModel#test(Collection)} method must be called.
 * The resulting accuracy will be logged.
 * In order to trigger the execute inference, {@link GenerativeQuestionAnsweringModel#generateAnswer(IQaTokenSequence)} method must be called.
 * </p>
 * <p>
 * The model uses a custom {@link GenerativeQuestionAnsweringModel.CustomListener} which allows to log different debug and execution data in order to show the
 * current training progress. It's also responsible for plotting the accuracy results.
 * </p>
 */
public class GenerativeQuestionAnsweringModel extends AbstractTransformerSameDiffModel<GenerativeQuestionAnsweringModel> {
    @Serial
    private static final long serialVersionUID = -5137157331190249821L;
    private static final Logger LOG = LoggerFactory.getLogger(GenerativeQuestionAnsweringModel.class);
    private static final String MODEL_FILE_NAME = "GenerativeQaTransformerBasedModel";
    protected static final String ENCODER_INPUT_MASKS_VAR_NAME = "encoderInputMasks";
    private static final String ENCODER_INPUT_VAR_NAME = "encoderInputTokenVocabIndices";
    protected static final String ENCODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME = "encoderPositionalEmbeddingsMatrix";
    private static final String ENCODER_TO_DECODER_INPUT_VAR_NAME = "concatenatedEncoderOutput";
    private static final String ENCODER_TO_DECODER_MASKS_INPUT_VAR_NAME = "concatenatedEncoderMasks";
    private static final String DECODER_INPUT_VAR_NAME = "decoderInputTokenVocabIndices";
    protected static final String DECODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME = "decoderPositionalEmbeddingsMatrix";
    protected static final String DECODER_SELF_ATTENTION_MASKS_VAR_NAME = "decoderSelfAttentionMasks";
    public static final String FLAT_TARGET_BATCH_POSITIONS_VAR_NAME = "FLAT_TARGET_BATCH_POSITIONS";
    public static final String FLAT_TARGET_TOKEN_VOCAB_INDICES_VAR_NAME = "FLAT_TARGET_TOKEN_VOCAB_INDICES";
    public static final String MODEL_OUTPUT_VAR_NAME = "finalLayerResult";
    public static final String ENCODER_ID = "generative_enc";
    public static final String DECODER_ID = "generative_dec";
    private static final Random random = new Random();

    private final int encoderSequenceLength;
    private final int encoderBatchSize;
    private final int epochsAmount;
    private final int encoderLayersAmount;
    private final int amountOfPassagesPerAnswer;
    private final boolean fixEncoderWeights;
    protected final int beamSearchWidth = 3;

    @Override
    protected String getModelFileName() {
        return MODEL_FILE_NAME;
    }

    @Override
    protected GenerativeQuestionAnsweringModel getInstance() {
        return this;
    }

    @Override
    protected TrainingConfig getTrainingConfig() {
        Adam adam = new Adam(learningRate, DEFAULT_ADAM_BETA1_MEAN_DECAY, beta2, DEFAULT_ADAM_EPSILON);
        if (optimizerSchedule != null) {
            adam.setLearningRateSchedule(optimizerSchedule);
        }

        String[] labels = new String[]{FLAT_TARGET_TOKEN_VOCAB_INDICES_VAR_NAME, FLAT_TARGET_BATCH_POSITIONS_VAR_NAME};
        String[] masks = new String[]{ENCODER_INPUT_MASKS_VAR_NAME, DECODER_SELF_ATTENTION_MASKS_VAR_NAME};
        return new TrainingConfig.Builder()
                .updater(adam)
                .dataSetFeatureMapping(ENCODER_INPUT_VAR_NAME, DECODER_INPUT_VAR_NAME)
                .dataSetLabelMapping(labels)
                .dataSetFeatureMaskMapping(masks)
                .build();
    }

    private GenerativeQuestionAnsweringModel(int encoderSequenceLength, int decoderSequenceLength, int encoderBatchSize, double beta2,
                                             int hiddenSize, int epochsAmount, int modelTestFrequency, WordPieceVocab vocab,
                                             int loggingFrequency, int modelSaveFrequency, double learningRate,
                                             int encoderLayersAmount, int decoderLayersAmount,
                                             int attentionHeadsAmount, int intermediateLayerSize, int amountOfPassagesPerAnswer,
                                             float dropoutRate, ISchedule optimizerSchedule, double labelSmoothing,
                                             boolean fixDecoderWeights, boolean fixEncoderWeights, boolean fixTokenEmbeddings) {
        super(encoderBatchSize / amountOfPassagesPerAnswer, decoderSequenceLength, modelTestFrequency, loggingFrequency,
                learningRate, hiddenSize, decoderLayersAmount, attentionHeadsAmount, intermediateLayerSize, labelSmoothing,
                fixDecoderWeights, fixTokenEmbeddings, modelSaveFrequency, beta2, dropoutRate);

        checkArgument(encoderBatchSize % amountOfPassagesPerAnswer == 0,
                "Encoder batch size %s must be evenly divisible by amount of passages per answer %s", encoderBatchSize,
                amountOfPassagesPerAnswer);
        this.encoderSequenceLength = encoderSequenceLength;
        this.encoderBatchSize = encoderBatchSize;
        this.epochsAmount = epochsAmount;
        this.encoderLayersAmount = encoderLayersAmount;
        this.amountOfPassagesPerAnswer = amountOfPassagesPerAnswer;
        this.fixEncoderWeights = fixEncoderWeights;
        this.optimizerSchedule = optimizerSchedule;
        this.vocab = vocab;

        buildCompleteSdGraph();
    }

    public void train(ArrayList<IQaTokenSequence> trainData, Consumer<GenerativeQuestionAnsweringModel> modelSaver,
                      int maxAmountOfTrainingSteps, Consumer<GenerativeQuestionAnsweringModel> tester)
            throws IOException, CloneNotSupportedException {
        requireNonNull(trainData, "Model can't be trained without data");
        String plotterName = format("%d_%d_%s", encoderBatchSize, batchSize, "generative-qa");
        CustomListener goodListener = new CustomListener(loggingFrequency, plotterName, vocab);
        int trainingStepsDone = 0;
        int epochsDone = 0;

        while (trainingStepsDone < maxAmountOfTrainingSteps && epochsDone < epochsAmount) {
            Instant epochStart = now();
            List<Integer> epochTrainDataIndices = range(0, trainData.size()).boxed().collect(toCollection(LinkedList::new));
            LOG.info("Starting the {} epoch expecting it to be complete in {} steps", getCounterEnding(epochsDone + 1),
                    format("%.1f", epochTrainDataIndices.size() / ((double) encoderBatchSize / amountOfPassagesPerAnswer)));

            while (!epochTrainDataIndices.isEmpty()) {
                sameDiff.setListeners(ImmutableSet.of(goodListener));
                int currentEncoderBatchIndex = 0;
                INDArray encoderBatchTokenVocabIndices = create(INT32, encoderBatchSize, encoderSequenceLength);
                INDArray encoderBatchSelfAttentionMasks = create(INT8, encoderBatchSize, encoderSequenceLength);
                INDArray decoderBatchTokenVocabIndices = create(INT32, batchSize, baseSequenceLength);
                INDArray decoderBatchSelfAttentionMasks = create(INT8, batchSize, baseSequenceLength);
                List<Integer> flatTargetBatchPositions = new LinkedList<>();
                List<Integer> flatTargetVocabIndices = new LinkedList<>();

                while (!epochTrainDataIndices.isEmpty() && currentEncoderBatchIndex < encoderBatchSize) {
                    var multipleContextsQaTokenSequence = pollNewSequence(trainData, epochTrainDataIndices).orElseThrow();
                    addEncoderSequenceTargetsAndMasks(currentEncoderBatchIndex, multipleContextsQaTokenSequence,
                            encoderBatchTokenVocabIndices, encoderBatchSelfAttentionMasks);

                    int decoderBatchIndex = currentEncoderBatchIndex / amountOfPassagesPerAnswer;
                    var answerTokens = stream(multipleContextsQaTokenSequence.getAnswerTokens())
                            .limit(baseSequenceLength)
                            .collect(toCollection(LinkedList::new));
                    if (answerTokens.size() < baseSequenceLength) {
                        answerTokens.add(WordPieceVocab.END_OF_SENTENCE_SYMBOL);
                    } else {
                        answerTokens.set(baseSequenceLength - 1, WordPieceVocab.END_OF_SENTENCE_SYMBOL);
                    }
                    addDecoderSequenceAndMasks(decoderBatchIndex, answerTokens, decoderBatchTokenVocabIndices,
                            decoderBatchSelfAttentionMasks);
                    addLabels(decoderBatchIndex, answerTokens, flatTargetBatchPositions, flatTargetVocabIndices);

                    currentEncoderBatchIndex += amountOfPassagesPerAnswer;
                }

                MultiDataSet multiDataSet = getMultiDataSet(encoderBatchTokenVocabIndices, decoderBatchTokenVocabIndices,
                        flatTargetBatchPositions, flatTargetVocabIndices, encoderBatchSelfAttentionMasks, decoderBatchSelfAttentionMasks);

                sameDiff.fit(new SingletonMultiDataSetIterator(multiDataSet), 1);
                if (goodListener.earlyTrainingStop) {
                    LOG.error("Early stop was requested after {} steps", trainingStepsDone);
                    break;
                }

                if (trainingStepsDone > 0 && trainingStepsDone % saveFrequencyInSteps == 0) {
                    LOG.info("Saving a model during epoch after {} step done", getCounterEnding(trainingStepsDone));
                    modelSaver.accept(this);
                }

                if ((trainingStepsDone + 1) % modelTestFrequency == 0) {
                    testDuringTraining(tester);
                }
                ++trainingStepsDone;
            }
            ++epochsDone;
            LOG.info("\n\n\n------------------- Done {} epoch in {} minutes\n\n", getCounterEnding(epochsDone),
                    getDurationInMinutes(epochStart));
            LOG.info("Wiping out updater states after epoch end");
            sameDiff.getUpdaterMap().forEach((s, gradientUpdater) ->
                    gradientUpdater.getState().clear());

            LOG.info("Saving a model after {} epoch", getCounterEnding(epochsDone));
            modelSaver.accept(this);
            System.gc();
        }

        LOG.info("Completed training after {} epochs and {} steps", epochsDone, trainingStepsDone);
    }

    public void test(Collection<IQaTokenSequence> multipleContextsQaTokenSequences) {
        List<Double> allAccuracies = new LinkedList<>();
        multipleContextsQaTokenSequences.forEach(multipleContextsQaTokenSequence -> {
            List<Double> answerAccuracies = new LinkedList<>();
            var answerTokens = stream(multipleContextsQaTokenSequence.getAnswerTokens()).collect(toCollection(LinkedList::new));
            if (answerTokens.size() < baseSequenceLength) {
                answerTokens.add(WordPieceVocab.END_OF_SENTENCE_SYMBOL);
            }

            INDArray out = getModelOutput(multipleContextsQaTokenSequence, answerTokens);
            List<String> actualAnswerTokens = new LinkedList<>();
            List<String> probabilities = new LinkedList<>();

            addSequenceAccuraciesAndActualPredictions(answerAccuracies, answerTokens, out, actualAnswerTokens, probabilities, true);
            allAccuracies.addAll(answerAccuracies);
            double accuracyAverage = answerAccuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            LOG.info("Question: " + String.join(" ", multipleContextsQaTokenSequence.getQuestionTokens()));
            LOG.info("Predicted Answer: {},  probabilities: {}", String.join(" ", actualAnswerTokens), probabilities);
            LOG.info("Actual Answer: " + String.join(" ", answerTokens));
            LOG.info("Accuracy: " + accuracyAverage);
            System.out.println("\n");
        });
        System.out.println("\n");
        LOG.info("Total accuracy: " + allAccuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0));
    }

    public Optional<String> generateAnswer(IQaTokenSequence multipleContextsQaTokenSequence) {
        List<PredictionResult> predictionResults = new LinkedList<>();

        INDArray primaryOutput = getModelOutput(multipleContextsQaTokenSequence, List.of(WordPieceVocab.PADDING_SYMBOL));
        getTopPredictions(primaryOutput, beamSearchWidth, 0).forEach((predictedToken, probability) -> {
            PredictionResult predictionResult = new PredictionResult();
            predictionResult.addPredictedToken(WordPieceVocab.PADDING_SYMBOL, 0);
            predictionResult.addPredictedToken(predictedToken, probability);
            predictionResults.add(predictionResult);
        });

        predictionResults.forEach(collectedPredictionResult -> {
            var alreadyPredictedTokens = collectedPredictionResult.getTokens();
            while (!alreadyPredictedTokens.getLast().equals(WordPieceVocab.END_OF_SENTENCE_SYMBOL) &&
                    alreadyPredictedTokens.size() < baseSequenceLength) {
                INDArray output = getModelOutput(multipleContextsQaTokenSequence, alreadyPredictedTokens);
                getTopPredictions(output, 1, alreadyPredictedTokens.size() - 1)
                        .forEach(collectedPredictionResult::addPredictedToken);
            }
        });

        List<String> topAnswers = predictionResults.stream()
                .sorted(Comparator.comparing(PredictionResult::getTotalProbability).reversed())
                .map(result -> reconstructAnswerFromTokens(result.getTokens()))
                .toList();
        LOG.info("Top answers: \n{}", topAnswers);

        return topAnswers.stream().findFirst();
    }

    public int getSequenceLengthWithoutSeparators() {
        return encoderSequenceLength - 3;
    }

    private MultiDataSet getMultiDataSet(INDArray batchEncoderTokenVocabIndices, INDArray batchDecoderTokenVocabIndices,
                                         List<Integer> flatTargetBatchPositions, List<Integer> flatTargetTokenVocabIndices,
                                         INDArray encoderBatchSelfAttentionMasks, INDArray decoderBatchSelfAttentionMasks) {
        var labels = new INDArray[]{createFromArray(flatTargetTokenVocabIndices.toArray(Integer[]::new)),
                createFromArray(flatTargetBatchPositions.toArray(Integer[]::new))};
        var features = new INDArray[]{batchEncoderTokenVocabIndices, batchDecoderTokenVocabIndices};

        return new org.nd4j.linalg.dataset.MultiDataSet(features, labels,
                new INDArray[]{encoderBatchSelfAttentionMasks, decoderBatchSelfAttentionMasks}, null);
    }

    protected void buildCompleteSdGraph() {
        var sd = SameDiff.create();
        sd.setTrainingConfig(getTrainingConfig());

        LOG.info("""                            
                Building a Generative Question-Answer SameDiff model with the following params:
                 - number of encoder blocks: {}
                 - number of decoder blocks: {}
                 - max encoder sequence length: {}
                 - max decoder sequence length: {}
                 - token embeddings and hidden layer size: {}
                 - vocab size: {}
                """, encoderLayersAmount, layersAmount, encoderSequenceLength, baseSequenceLength, hiddenSize, vocab.size());

        // Token embeddings
        var tokenEmbeddingsMatrix = sd.var("tokenEmbeddingsMatrix",
                new XavierInitScheme('c', vocab.size(), hiddenSize), FLOAT, vocab.size(), hiddenSize);
        var encoderInputMasks = sd.placeHolder(ENCODER_INPUT_MASKS_VAR_NAME, FLOAT, -1, encoderSequenceLength);
        var decoderSelfAttentionMasks = sd.placeHolder(DECODER_SELF_ATTENTION_MASKS_VAR_NAME, FLOAT, -1, baseSequenceLength);
        var encoderInputTokenVocabIndices = sd.placeHolder(ENCODER_INPUT_VAR_NAME, INT32, -1, encoderSequenceLength);
        var decoderInputTokenVocabIndices = sd.placeHolder(DECODER_INPUT_VAR_NAME, INT32, -1, baseSequenceLength);
        var encoderPositionalEmbeddingsMatrix = sd.var(ENCODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME,
                initializePositionalEmbeddings(encoderSequenceLength)).convertToConstant();
        var decoderPositionalEmbeddingsMatrix = sd.var(DECODER_POSITIONAL_EMBEDDINGS_MATRIX_VAR_NAME,
                initializePositionalEmbeddings(baseSequenceLength)).convertToConstant();

        if (fixTokenEmbeddings) {
            tokenEmbeddingsMatrix.convertToConstant();
            encoderPositionalEmbeddingsMatrix.convertToConstant();
            decoderPositionalEmbeddingsMatrix.convertToConstant();
        }

        // Looking up token (word) and segment embeddings
        var batchEncoderInputTokenEmbeddings =
                gather(sd, "batchEncoderInputTokenEmbeddings", tokenEmbeddingsMatrix, encoderInputTokenVocabIndices, 0);
        var decoderBatchInputTokenEmbeddings =
                gather(sd, "decoderBatchInputTokenEmbeddings", tokenEmbeddingsMatrix, decoderInputTokenVocabIndices, 0);

        // Building encoder's graph
        var transformerEncoderGraphGenerator = getTransformerEncoderGraphGenerator();
        // [encoderBatchSize*sequenceLength, hiddenSize]
        var encoderOutput = transformerEncoderGraphGenerator.generateGraph(sd, encoderInputMasks,
                 batchEncoderInputTokenEmbeddings, encoderPositionalEmbeddingsMatrix);

        // Building decoder's graph
        var passagesPerAnswer = sd.constant("passagesPerAnswer", amountOfPassagesPerAnswer);
        var decoderBatchSize = getDimensionSize(encoderInputMasks, 0).div("decoderBatchSize", passagesPerAnswer);
        var hiddenSizeSd = getOrCreateConstant(sd, "hiddenSizeSd", hiddenSize);
        var concatenatedEncoderSequenceLength = passagesPerAnswer.mul(encoderSequenceLength);
        var concatenatedEncoderShape = sd.stack("concatenatedEncoderShape", 0, decoderBatchSize, concatenatedEncoderSequenceLength,
                hiddenSizeSd);
        var concatenatedEncoderOutput = encoderOutput.reshape(concatenatedEncoderShape).rename(ENCODER_TO_DECODER_INPUT_VAR_NAME);
        var concatenatedEncoderMasksShape =
                sd.stack("concatenatedEncoderMasksShape", 0, decoderBatchSize, concatenatedEncoderSequenceLength);
        var concatenatedEncoderMasks =
                encoderInputMasks.reshape(concatenatedEncoderMasksShape).rename(ENCODER_TO_DECODER_MASKS_INPUT_VAR_NAME);

        var questionBatchIndices = getPositionsRange(sd, 0, encoderSequenceLength);
        var contextsBatchIndices = getPositionsRange(sd, encoderSequenceLength, concatenatedEncoderSequenceLength);
        var encoderBatchQuestionTokensOutput = gather(sd, "encoderBatchQuestionTokensOutput", concatenatedEncoderOutput,
                questionBatchIndices, 1);
        var encoderBatchContextTokensOutput = gather(sd, "encoderBatchContextTokensOutput", concatenatedEncoderOutput,
                contextsBatchIndices, 1);
        var encoderBatchQuestionMasks = gather(sd, "encoderBatchQuestionMasks", concatenatedEncoderMasks, questionBatchIndices, 1);
        var encoderBatchContextsMasks = gather(sd, "encoderBatchContextMasks", concatenatedEncoderMasks, contextsBatchIndices, 1);

        var encoderDecoderGraphGenerator = getTransformerQaDecoderGraphGenerator();
        var decoderOutput = encoderDecoderGraphGenerator.generateGraph(sd, decoderBatchInputTokenEmbeddings, decoderSelfAttentionMasks,
                encoderBatchQuestionTokensOutput, encoderBatchQuestionMasks, encoderBatchContextTokensOutput,
                encoderBatchContextsMasks, decoderPositionalEmbeddingsMatrix,
                sd.constant("causalMasksTemplate", createFromArray(createCausalMasksTemplate(baseSequenceLength))));

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
    private TransformerEncoderGraphGenerator getTransformerEncoderGraphGenerator() {
        return new TransformerEncoderGraphGenerator.Builder(ENCODER_ID)
                .withAttentionHeadsAmount(this.attentionHeadsAmount)
                .withLayersAmount(this.encoderLayersAmount)
                .withHiddenSize(hiddenSize)
                .withIntermediateLayerSize(this.intermediateLayerSize)
                .withSequenceLength(this.encoderSequenceLength)
                .withFixWeights(fixEncoderWeights)
                .withDropoutRate(dropout)
                .build();
    }

    @Nonnull
    private TransformerQaDecoderGraphGenerator getTransformerQaDecoderGraphGenerator() {
        return new TransformerQaDecoderGraphGenerator.Builder(DECODER_ID)
                .withAttentionHeadsAmount(this.attentionHeadsAmount)
                .withLayersAmount(this.layersAmount)
                .withHiddenSize(hiddenSize)
                .withIntermediateLayerSize(this.intermediateLayerSize)
                .withSequenceLength(this.baseSequenceLength)
                .withFixWeights(fixWeights)
                .withDropoutRate(dropout)
                .build();
    }

    private void addEncoderSequenceTargetsAndMasks(int lastBatchIndex, IQaTokenSequence multipleContextsQaTokenSequence,
                                                   INDArray batchEncoderTokenVocabIndices, INDArray encoderBatchFeatureMasks) {
        var questionTokens = multipleContextsQaTokenSequence.getQuestionTokens();
        var contextTokensList = multipleContextsQaTokenSequence.getShuffledContextTokensList();

        // First go the question tokens
        addEncoderSequenceTokenIndicesAndMasks(batchEncoderTokenVocabIndices, encoderBatchFeatureMasks, questionTokens, lastBatchIndex);

        // Now the passages themselves
        int passagesAmountToAdd = min(contextTokensList.size(), amountOfPassagesPerAnswer - 1);
        range(0, passagesAmountToAdd)
                .forEach(contextIndex -> {
                    String[] contextTokens = contextTokensList.get(contextIndex);
                    addEncoderSequenceTokenIndicesAndMasks(batchEncoderTokenVocabIndices, encoderBatchFeatureMasks, contextTokens,
                            contextIndex + lastBatchIndex + 1);
                });

        // Now padding the remaining positions
        range(passagesAmountToAdd + 1, amountOfPassagesPerAnswer)
                .forEach(contextIndex -> addEncoderPaddingSequences(batchEncoderTokenVocabIndices, encoderBatchFeatureMasks,
                        contextIndex + lastBatchIndex));
    }

    private void addEncoderPaddingSequences(INDArray batchEncoderTokenVocabIndices, INDArray encoderBatchFeatureMasks, int contextIndex) {
        batchEncoderTokenVocabIndices.putRow(contextIndex, valueArrayOf(new long[]{encoderSequenceLength},
                vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow(), INT32));
        encoderBatchFeatureMasks.putRow(contextIndex, zeros(new int[]{encoderSequenceLength}, INT8));
    }

    private void addEncoderSequenceTokenIndicesAndMasks(INDArray batchEncoderTokenVocabIndices, INDArray encoderBatchFeatureMasks,
                                                        String[] tokens, int contextIndex) {
        INDArray encoderSequenceFeatureMasks = create(INT8, encoderSequenceLength);
        INDArray sequenceTokenVocabIndices = create(INT32, encoderSequenceLength);

        range(0, min(tokens.length, encoderSequenceLength))
                .forEach(questionTokenPosition -> {
                    Integer tokenVocabIndex = vocab.getTokenIndex(tokens[questionTokenPosition]).orElseThrow();
                    sequenceTokenVocabIndices.putScalar(questionTokenPosition, tokenVocabIndex);
                    encoderSequenceFeatureMasks.putScalar(questionTokenPosition, 1);
                });
        for (int i = tokens.length; i < encoderSequenceLength; i++) {
            sequenceTokenVocabIndices.putScalar(i, vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow());
            encoderSequenceFeatureMasks.putScalar(i, 0);
        }

        batchEncoderTokenVocabIndices.putRow(contextIndex, sequenceTokenVocabIndices);
        encoderBatchFeatureMasks.putRow(contextIndex, encoderSequenceFeatureMasks);
    }

    private void addDecoderSequenceAndMasks(int batchIndex, List<String> answerTokens, INDArray decoderBatchTokenVocabIndices,
                                            INDArray decoderBatchFeatureMasks) {
        var decoderInputTokens = new LinkedList<>(answerTokens);
        decoderInputTokens.addFirst(WordPieceVocab.PADDING_SYMBOL);
        decoderInputTokens.removeLast();
        INDArray sequenceFeatureMasks = create(INT8, baseSequenceLength);
        INDArray sequenceTokenVocabIndices = create(INT32, baseSequenceLength);

        range(0, decoderInputTokens.size())
                .limit(baseSequenceLength)
                .forEach(index -> {
                    sequenceTokenVocabIndices.putScalar(index, vocab.getTokenIndex(decoderInputTokens.get(index)).orElseThrow());
                    sequenceFeatureMasks.putScalar(index, 1);
                });
        if (decoderInputTokens.size() < baseSequenceLength) {
            range(decoderInputTokens.size(), baseSequenceLength)
                    .forEach(index -> {
                        sequenceTokenVocabIndices.putScalar(index, vocab.getTokenIndex(WordPieceVocab.PADDING_SYMBOL).orElseThrow());
                        sequenceFeatureMasks.putScalar(index, 0);
                    });
        }

        decoderBatchFeatureMasks.putRow(batchIndex, sequenceFeatureMasks);
        decoderBatchTokenVocabIndices.putRow(batchIndex, sequenceTokenVocabIndices);
    }

    private void addLabels(int batchIndex, List<String> answerTokens, List<Integer> flatTargetBatchPositions,
                           List<Integer> flatTargetVocabIndices) {
        answerTokens.stream()
                .limit(baseSequenceLength)
                .map(answerToken -> vocab.getTokenIndex(answerToken).orElseThrow())
                .forEach(flatTargetVocabIndices::add);

        range(0, answerTokens.size())
                .limit(baseSequenceLength)
                .forEach(answerTokenIndex -> flatTargetBatchPositions.add(batchIndex * baseSequenceLength + answerTokenIndex));
    }

    private Optional<IQaTokenSequence> pollNewSequence(List<IQaTokenSequence> allSequences,
                                                                      List<Integer> availableIndices) {
        if (!availableIndices.isEmpty()) {
            Optional<Integer> sequenceIndexOptional = random.ints(0, availableIndices.size())
                    .distinct()
                    .mapToObj(availableIndices::get)
                    .findAny();
            sequenceIndexOptional.ifPresent(availableIndices::remove);

            return sequenceIndexOptional.map(allSequences::get);
        } else {
            return empty();
        }
    }

    private void addSequenceAccuraciesAndActualPredictions(List<Double> answerAccuraciesCollector, LinkedList<String> answerTokens,
                                                           INDArray out, List<String> actualPredictionTokensCollector,
                                                           List<String> probabilitiesCollector, boolean ignoreEos) {
        var predictionLogits = Nd4j.nn().softmax(out, 1);
        double[][] predictionLogitsMatrix = predictionLogits.toDoubleMatrix();
        boolean endOfSentenceReached = false;
        for (int i = 0; i < answerTokens.size(); i++) {
            if (predictionLogitsMatrix.length > i && !endOfSentenceReached) {
                double[] softmaxResults = predictionLogitsMatrix[i];
                List<String> predictions = range(0, softmaxResults.length)
                        .mapToObj(index -> Pair.of(index, softmaxResults[index]))
                        .sorted(Entry.<Integer, Double>comparingByValue().reversed())
                        .limit(3)
                        .map(valueByIndex -> vocab.getTokenByIndex(valueByIndex.getKey()).orElse(""))
                        .toList();
                var topPrediction = predictions.get(0);
                if (topPrediction.equals(WordPieceVocab.END_OF_SENTENCE_SYMBOL)) {
                    endOfSentenceReached = true;
                }
                actualPredictionTokensCollector.add(topPrediction);
                probabilitiesCollector.add(format("%s <%.4f> ", topPrediction, stream(softmaxResults).max().orElse(0)));
                boolean accuracyCalculationNeeded = !(endOfSentenceReached && ignoreEos);
                if (accuracyCalculationNeeded) {
                    Map<String, Integer> predictionsByPosition = range(0, predictions.size())
                            .boxed()
                            .collect(toMap(predictions::get, identity()));
                    double accuracy = ofNullable(predictionsByPosition.get(answerTokens.get(i)))
                            .map(predictionPosition -> 100 / ((double) predictionPosition + 1)).orElse(0d);
                    answerAccuraciesCollector.add(accuracy);
                }
            } else {
                answerAccuraciesCollector.add(0d);
            }
        }
    }

    private INDArray getModelOutput(IQaTokenSequence multipleContextsQaTokenSequence, List<String> answerTokens) {
        var sd = sameDiff;
        INDArray encoderBatchFeatureMasks = create(INT8, amountOfPassagesPerAnswer, encoderSequenceLength);
        INDArray encoderBatchTokenVocabIndices = create(INT32, amountOfPassagesPerAnswer, encoderSequenceLength);
        addEncoderSequenceTargetsAndMasks(0, multipleContextsQaTokenSequence, encoderBatchTokenVocabIndices, encoderBatchFeatureMasks);
        sd.getVariable(ENCODER_INPUT_MASKS_VAR_NAME).setArray(encoderBatchFeatureMasks);
        sd.getVariable(ENCODER_INPUT_VAR_NAME).setArray(encoderBatchTokenVocabIndices);

        INDArray decoderBatchTokenVocabIndices = create(INT32, 1, baseSequenceLength);
        INDArray decoderBatchFeatureMasks = create(INT8, 1, baseSequenceLength);
        addDecoderSequenceAndMasks(0, answerTokens, decoderBatchTokenVocabIndices, decoderBatchFeatureMasks);
        sd.getVariable(DECODER_INPUT_VAR_NAME).setArray(decoderBatchTokenVocabIndices);
        sd.getVariable(DECODER_SELF_ATTENTION_MASKS_VAR_NAME).setArray(decoderBatchFeatureMasks);

        List<Integer> flatTargetBatchPositions = new LinkedList<>();
        addLabels(0, answerTokens, flatTargetBatchPositions, new LinkedList<>());
        sd.getVariable(FLAT_TARGET_BATCH_POSITIONS_VAR_NAME).setArray(createFromArray(flatTargetBatchPositions.toArray(Integer[]::new)));

        return sd.getVariable(MODEL_OUTPUT_VAR_NAME).eval();
    }

    private Map<String, Double> getTopPredictions(INDArray modelOutput, int limit, int sequencePositionToPredict) {
        double[] predictionLogits = Nd4j.nn().softmax(modelOutput, 1).slice(sequencePositionToPredict).toDoubleVector();
        return range(0, predictionLogits.length)
                .mapToObj(index -> Pair.of(index, predictionLogits[index]))
                .sorted(Entry.<Integer, Double>comparingByValue().reversed())
                .limit(limit)
                .collect(toMap(probabilityByTokenIndex -> vocab.getTokenByIndex(probabilityByTokenIndex.getKey()).orElse(""),
                        Entry::getValue));
    }

    private String reconstructAnswerFromTokens(List<String> tokens) {
        return reconstructFromTokens(tokens);
    }

    private static class PredictionResult {
        private final LinkedList<String> tokens = new LinkedList<>();
        private double totalProbability;

        public void addPredictedToken(String token, double probability) {
            tokens.add(requireNonNull(token));
            totalProbability += probability;
        }

        public LinkedList<String> getTokens() {
            return tokens;
        }

        public double getTotalProbability() {
            return totalProbability;
        }
    }

    /**
     * A basic custom listener that logs the useful info (like current learning rate, accuracy during training and testing, loss)
     * and does some preventive measures (like zeroing out NaNs in weight updates, early stopping etc.)
     */
    public static class CustomListener extends BaseListener {
        private boolean earlyTrainingStop;
        private static final Condition nanCondition = Conditions.isNan();
        private final int logFrequency;
        private final List<Double> latestAccuracies = new LinkedList<>();
        private final List<Double> latestLosses = new LinkedList<>();
        private Instant start = now();
        private final ConvergenceChartPlotter plotter;
        private static final String ACCURACY = "Accuracy";
        private static final String LOSS = "Loss";
        private final WordPieceVocab vocab;

        public CustomListener(int logFrequency, String plotterName, WordPieceVocab vocab) {
            this.logFrequency = logFrequency;
            plotter = newChartPlotter(plotterName, List.of(LOSS, ACCURACY));
            this.vocab = vocab;
        }

        @Override
        public void epochStart(SameDiff sd, At at) {
            super.epochStart(sd, at);
            start = now();
        }

        @Override
        public void activationAvailable(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName, INDArray activation) {
            /*if (varName.contains("finalLayerResult")) {
                int i = 0;
            }*/
        }

        @Override
        public boolean isActive(Operation operation) {
            return true;
        }

        @Override
        public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
            boolean isNan = BooleanIndexing.or(update, nanCondition);
            if (isNan) {
                if (timeToLog(at.iteration() + 1)) {
                    LOG.warn("NaN in gradient update for '{}'. Zeroing out NaN values in order to avoid destroying weights",
                            v.getName());
                }
                replaceWhere(update, 0.0, nanCondition);
            }
        }

        @Override
        public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext,
                                INDArray[] outputs) {
            if (batch != null && op.getOutputsOfOp().contains("modelPredictionLogits")) {
                Pair<Double, Double> accuracyByLoss = getAccuracyByLoss(batch, outputs[0]);
                latestLosses.add(accuracyByLoss.getKey());
                latestAccuracies.add(accuracyByLoss.getValue());
                if (timeToLog(at.iteration() + 1)) {
                    double averageLoss = abs(latestLosses.stream().mapToDouble(Double::doubleValue).average().orElse(0));
                    double averageAccuracy = latestAccuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                    LOG.info("After {} iteration:   Loss {},    Accuracy {}",
                            getCounterEnding(at.iteration() + 1), format("%.5f", averageLoss), format("%.1f", averageAccuracy));
                    plotter.addData(LOSS, at.iteration() + 1, averageLoss);
                    plotter.addData(ACCURACY, at.iteration() + 1, averageAccuracy);
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
                var label = labels.getInt(i);
                if (vocab.getTokenIndex(WordPieceVocab.END_OF_SENTENCE_SYMBOL).orElseThrow() != label) {
                    predictionMatchStates.add(actualPredictions[i] == label);
                }
            }
            long truePredictions = predictionMatchStates.stream()
                    .filter(aBoolean -> aBoolean)
                    .count();
            double accuracy = truePredictions / (double) predictionMatchStates.size() * 100;

            return Pair.of(latestLoss, accuracy);
        }
    }


    public static final class Builder extends AbstractTransformerSameDiffModel.Builder<Builder, GenerativeQuestionAnsweringModel> {
        private int encoderSequenceLength = 256;
        private int encoderBatchSize = 500;
        private int epochsAmount = 10;
        private int encoderLayersAmount = 4;
        private int amountOfPassagesPerAnswer = 20;
        private boolean fixEncoderWeights = true;
        private WarmupWithPolyDecaySchedule optimizerSchedule;

        public Builder(WordPieceVocab vocab) {
            super(vocab);
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public Builder withOptimizerSchedule(WarmupWithPolyDecaySchedule optimizerSchedule) {
            this.optimizerSchedule = optimizerSchedule;
            return this;
        }

        public Builder withEncoderSequenceLength(int encoderSequenceLength) {
            this.encoderSequenceLength = encoderSequenceLength;
            return this;
        }

        public Builder withEncoderBatchSize(int encoderBatchSize) {
            this.encoderBatchSize = encoderBatchSize;
            return this;
        }

        public Builder withEpochsAmount(int epochsAmount) {
            this.epochsAmount = epochsAmount;
            return this;
        }

        public Builder withEncoderLayersAmount(int encoderLayersAmount) {
            this.encoderLayersAmount = encoderLayersAmount;
            return this;
        }

        public Builder withAmountOfPassagesPerAnswer(int amountOfPassagesPerAnswer) {
            this.amountOfPassagesPerAnswer = amountOfPassagesPerAnswer;
            return this;
        }

        public Builder withFixedEncoderWeights(boolean fixEncoderWeights) {
            this.fixEncoderWeights = fixEncoderWeights;
            return this;
        }

        public GenerativeQuestionAnsweringModel build() {


            return new GenerativeQuestionAnsweringModel(encoderSequenceLength, sequenceLength, encoderBatchSize, beta2, hiddenSize,
                    epochsAmount, modelTestFrequency, vocab, loggingFrequency, saveFrequencyInSteps, learningRate, encoderLayersAmount,
                    layersAmount, attentionHeadsAmount, intermediateLayerSize, amountOfPassagesPerAnswer, dropout, optimizerSchedule,
                    labelSmoothing, fixWeights, fixEncoderWeights, fixTokenEmbeddings);
        }
    }
}