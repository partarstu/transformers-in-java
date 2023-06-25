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

package org.tarik.core.network.models.transformer;

import com.google.gson.Gson;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.generational.GenerativeTransformerSdModel;
import org.tarik.core.network.models.transformer.mlm.MlmTransformerSdModel;
import org.tarik.core.vocab.WordPieceVocab;

import java.io.*;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import static java.lang.Math.*;
import static java.lang.String.format;
import static java.nio.file.Files.exists;
import static java.nio.file.Files.newOutputStream;
import static java.time.Instant.now;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.joining;
import static java.util.stream.IntStream.range;
import static org.nd4j.autodiff.samediff.VariableType.CONSTANT;
import static org.nd4j.autodiff.samediff.VariableType.VARIABLE;
import static org.nd4j.common.base.Preconditions.checkState;
import static org.nd4j.linalg.factory.Nd4j.createFromArray;
import static org.nd4j.linalg.learning.config.Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY;
import static org.nd4j.linalg.learning.config.Adam.DEFAULT_ADAM_EPSILON;
import static org.tarik.utils.CommonUtils.getDurationInSeconds;

/**
 * Generic Transformers-based Model implemented using {@link SameDiff}. Contains the common functionality like generating attention
 * masks, positional embeddings, saving and loading the model and its weights.
 *
 * @param <T> the type of the child model
 */
public abstract class AbstractTransformerSameDiffModel<T extends AbstractTransformerSameDiffModel<T>> implements Serializable, Cloneable {
    @Serial
    private static final long serialVersionUID = 1500414633801322975L;
    private static final Logger LOG = LoggerFactory.getLogger(AbstractTransformerSameDiffModel.class);
    protected static final String TOKEN_EMBEDDINGS_MATRIX_VAR_NAME = "tokenEmbeddingsMatrix";
    protected static final String SAME_DIFF_FILE_NAME = "SameDiff";
    protected static final Gson gson = new Gson();
    protected transient SameDiff sameDiff;
    protected ISchedule optimizerSchedule;
    protected WordPieceVocab vocab;
    protected int epochsDone;
    protected final int batchSize;
    protected final int baseSequenceLength;
    protected final int modelTestFrequency;
    protected final int loggingFrequency;
    protected final double learningRate;
    protected final int hiddenSize;
    protected final int layersAmount;
    protected final int attentionHeadsAmount;
    protected final int intermediateLayerSize;
    protected final double labelSmoothing;
    protected final boolean fixWeights;
    protected final boolean fixTokenEmbeddings;
    protected final int saveFrequencyInSteps;
    protected final double beta2;
    protected final float dropout;

    public AbstractTransformerSameDiffModel(int batchSize, int contextSequenceLength, int modelTestFrequency, int loggingFrequency,
                                            double learningRate, int hiddenSize, int layersAmount, int attentionHeadsAmount,
                                            int intermediateLayerSize, double labelSmoothing, boolean fixWeights,
                                            boolean fixTokenEmbeddings, int saveFrequencyInSteps, double beta2, float dropout) {
        this.batchSize = batchSize;
        this.baseSequenceLength = contextSequenceLength;
        this.modelTestFrequency = modelTestFrequency;
        this.loggingFrequency = loggingFrequency;
        this.learningRate = learningRate;
        this.hiddenSize = hiddenSize;
        this.layersAmount = layersAmount;
        this.attentionHeadsAmount = attentionHeadsAmount;
        this.intermediateLayerSize = intermediateLayerSize;
        this.labelSmoothing = labelSmoothing;
        this.fixWeights = fixWeights;
        this.fixTokenEmbeddings = fixTokenEmbeddings;
        this.saveFrequencyInSteps = saveFrequencyInSteps;
        this.beta2 = beta2;
        this.dropout = dropout;
    }

    protected abstract String getModelFileName();

    protected abstract T getInstance();

    protected abstract void buildCompleteSdGraph();

    protected abstract TrainingConfig getTrainingConfig();

    public WordPieceVocab getVocab() {
        return vocab;
    }

    /**
     * Provides the total amount of all model parameters (weights)
     *
     * @return the amount
     */
    public long getTotalParamsAmount() {
        return sameDiff.variables().stream()
                .filter(variable -> variable.getVariableType() == VARIABLE)
                .map(SDVariable::getShape)
                .filter(Objects::nonNull)
                .map(shape -> Arrays.stream(shape).reduce(1, Math::multiplyExact))
                .mapToLong(Long::longValue)
                .sum();
    }

    @Override
    public T clone() {
        try {
            T clone = (T) super.clone();
            clone.vocab = this.vocab;
            clone.sameDiff = this.sameDiff.dup();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Saves the model data into the ZIP archive. Two files are placed there - the model file itself (serialized Java class) and the
     * SameDiff data file. The stored during training updater's internal state will be saved by default.
     * @param path destination file path
     * @throws IOException in case any file-related issues arise
     */
    public void save(Path path) throws IOException {
        save(path, true);
    }

    /**
     * Saves the model data into the ZIP archive. Two files are placed there - the model file itself (serialized Java class) and the
     * SameDiff data file.
     * @param path destination file path
     * @param saveUpdater whether the stored during training updater's internal state should be saved
     * @throws IOException in case any file-related issues arise
     */
    public void save(Path path, boolean saveUpdater) throws IOException {
        ZipEntry modelFile = new ZipEntry(getModelFileName());
        ZipEntry sameDiffFile = new ZipEntry(SAME_DIFF_FILE_NAME);

        try (ZipOutputStream zipFile = new ZipOutputStream(new BufferedOutputStream(newOutputStream(path)))) {
            zipFile.putNextEntry(modelFile);
            try (ObjectOutputStream zipSerializationStream = new ObjectOutputStream(zipFile)) {
                zipSerializationStream.writeObject(this);
                zipFile.flush();
                zipFile.putNextEntry(sameDiffFile);
                sameDiff.save(zipFile, saveUpdater);
                zipFile.flush();
            }
        }
    }

    /**
     * Loads the model from the file to which it was previously saved using
     * {@link AbstractTransformerSameDiffModel#save(Path, boolean)} method. Expects exactly the same file structure as the one created
     * during saving process. If any issues related to the loading of the models from the file system arise - an error will be logged.
     * The saved SameDiff file won't replace the existing SameDiff instance. Instead, the array values of all constants and variables will
     * be loaded and will replace the current ones. This way of loading the data is not safe but allows to load the model's weights even
     * if the execution graph has been changed. It's quite useful in the course of model's development when the graph is being constantly
     * changed but the model has already been partially trained so that data is not lost.
     *
     * @param zipFilePath path to the ZIP file where the model file and the flat buffers file are stored
     * @param loadUpdater whether the stored during training updater's internal state should be loaded (is useful if the model was
     *                    saved during training and is being loaded to continue training where it's left)
     * @return the loaded model
     * @throws IOException in case any file-related issues arise
     */
    public Optional<T> loadModelDataFromFile(Path zipFilePath, boolean loadUpdater) throws IOException {
        checkState(exists(requireNonNull(zipFilePath)), "File %s doesn't exist", zipFilePath);
        T loadedModel = null;
        try (ZipFile zipFile = new ZipFile(zipFilePath.toFile())) {
            var zippedFilesAmount = zipFile.stream().count();
            var allZippedFileNames = zipFile.stream().map(ZipEntry::getName).collect(joining(", ", "'", "'"));
            checkState(zippedFilesAmount == 2,
                    "Exactly 2 files are expected to be in '%s' archive: a model itself and the SameDiff file. Actual files: %s",
                    zipFilePath, allZippedFileNames);
            checkState(zipFile.stream().map(ZipEntry::getName).anyMatch(SAME_DIFF_FILE_NAME::equals),
                    "'%s' archive doesn't have SameDiff file with name '%s'. Actual files: %s",
                    zipFilePath, SAME_DIFF_FILE_NAME, allZippedFileNames);
            ZipEntry modelFile = zipFile.stream()
                    .filter(zipEntry -> !zipEntry.getName().equals(SAME_DIFF_FILE_NAME))
                    .findFirst()
                    .orElseThrow(() -> new IllegalStateException(format("'%s' archive doesn't have the model file. Actual files: %s",
                            zipFilePath, allZippedFileNames)));
            LOG.info("Found the file '{}' inside '{}' model archive. Using it as a model file", modelFile.getName(), zipFilePath);

            try (ObjectInputStream modelSerialStream = new ObjectInputStream(zipFile.getInputStream(modelFile))) {
                Object deserializedModel = modelSerialStream.readObject();
                requireNonNull(deserializedModel,
                        format("Model file %s in ZIP file %s contains no data", getModelFileName(), zipFilePath));
                if (deserializedModel.getClass().equals(getInstance().getClass())) {
                    loadedModel = (T) getInstance().getClass().cast(deserializedModel);
                    this.vocab = ofNullable(loadedModel.vocab).orElse(this.vocab);
                    LOG.info("Using the token vocabulary from the loaded model");
                    if (loadUpdater) {
                        this.optimizerSchedule = ofNullable(loadedModel.optimizerSchedule).orElse(this.optimizerSchedule);
                        if (this.optimizerSchedule != null) {
                            this.epochsDone = loadedModel.epochsDone;
                            LOG.info("Using an optimizer schedule from the loaded model. Epochs done: {}, steps done: {}", epochsDone,
                                    loadedModel.getTrainingConfig().getIterationCount());
                        }
                    }
                } else {
                    LOG.error("Model couldn't be loaded because the object saved in file {} within ZIP file {} is not of type {}",
                            getModelFileName(), zipFilePath, getInstance().getClass());
                }
            } catch (ObjectStreamException | ClassNotFoundException e) {
                // Something went wrong during model load - could be a different version of the model or something else, so just logging
                // that info and hoping that the vocab which was used for SameDiff creation and training is the same, otherwise there
                // will be a lot of problems
                loadUpdater = false;
                LOG.error("Model couldn't be deserialized from zip file entry {}. The model's SameDiff file however will still be loaded",
                        modelFile);
            }

            ZipEntry sameDiffFile = zipFile.getEntry(SAME_DIFF_FILE_NAME);
            if (sameDiffFile != null) {
                loadSameDiffVariablesAndUpdaters(zipFilePath, zipFile, sameDiffFile, loadUpdater);
            } else {
                LOG.error("SameDiff file {} is not present in ZIP file {}. SameDiff loading has been skipped",
                        SAME_DIFF_FILE_NAME, zipFilePath);
            }
        }
        return ofNullable(loadedModel);
    }

    protected synchronized void saveModelDuringTraining(BiConsumer<T, Long> incompleteEpochModelSaver, int processedStepsAmount) {
        LOG.info("Saving a model after {} steps and incomplete mini-epochs round", processedStepsAmount);
        incompleteEpochModelSaver.accept(getInstance(), 0L);
    }

    protected INDArray getTokenSelfAttentionMasks(Collection<Integer> positionsToIgnore, int lastTokenIndex) {
        List<Integer> attentionMasks = new LinkedList<>();
        int firstPaddingPosition = min(lastTokenIndex + 1, baseSequenceLength);
        range(0, firstPaddingPosition)
                .map(position -> positionsToIgnore.contains(position) ? 0 : 1)
                .forEach(attentionMasks::add);
        range(firstPaddingPosition, baseSequenceLength)
                .forEach(position -> attentionMasks.add(0));

        return createFromArray(attentionMasks.toArray(Integer[]::new));
    }

    protected int[][] createCausalMasksTemplate(int sequenceLength) {
        var masks = new int[sequenceLength][sequenceLength];
        range(0, sequenceLength)
                .forEach(index -> {
                    int[] row = new int[sequenceLength];
                    Arrays.fill(row, 0, index + 1, 1);
                    masks[index] = row;
                });
        return masks;
    }

    protected INDArray getTokenCausalSelfAttentionMasks(int lastTokenIndex) {
        int[][] attentionCausalMasks = createCausalMasksTemplate(baseSequenceLength);
        int firstPaddingPosition = min(lastTokenIndex + 1, baseSequenceLength);
        range(firstPaddingPosition, baseSequenceLength)
                .forEach(paddingPosition -> attentionCausalMasks[paddingPosition] = new int[baseSequenceLength]);

        return Nd4j.expandDims(createFromArray(attentionCausalMasks), 0);
    }

    protected INDArray initializePositionalEmbeddings(int sequenceLength) {
        double[][] positionalEmbeddings = new double[sequenceLength][hiddenSize];
        for (int position = 0; position < positionalEmbeddings.length; position++) {
            for (int j = 0; j < hiddenSize / 2; j++) {
                positionalEmbeddings[position][j * 2] =
                        sin(position / pow(10000, (2 * j / (double) hiddenSize)));
                positionalEmbeddings[position][j * 2 + 1] =
                        cos(position / pow(10000, (2 * j / (double) hiddenSize)));
            }
        }
        return createFromArray(positionalEmbeddings);
    }

    protected IUpdater getUpdater() {
        Adam adam = new Adam(learningRate, DEFAULT_ADAM_BETA1_MEAN_DECAY, beta2, DEFAULT_ADAM_EPSILON);
        if (optimizerSchedule != null) {
            adam.setLearningRateSchedule(optimizerSchedule);
        }
        return adam;
    }

    protected void testDuringTraining(Consumer<T> tester) {
        Instant testingStart = now();
        tester.accept(getInstance());
        LOG.info("Testing done in {} sec", getDurationInSeconds(testingStart));
    }

    private void loadSameDiffVariablesAndUpdaters(Path zipFilePath, ZipFile zipFile, ZipEntry sameDiffFile, boolean loadUpdater)
            throws IOException {
        try (InputStream sameDiffInputStream = zipFile.getInputStream(sameDiffFile)) {
            SameDiff loadedSameDiff = SameDiff.load(sameDiffInputStream, loadUpdater);
            requireNonNull(loadedSameDiff,
                    format("SameDiff file %s in ZIP file %s contains no data", SAME_DIFF_FILE_NAME, zipFilePath));

            if (loadUpdater && loadedSameDiff.getTrainingConfig() != null) {
                LOG.info("Using updater from the loaded model");
                this.sameDiff.setTrainingConfig(loadedSameDiff.getTrainingConfig());
                ofNullable(this.sameDiff.getTrainingConfig().getUpdater())
                        .ifPresent(updater -> updater.setLrAndSchedule(this.learningRate, this.optimizerSchedule));
            }

            LOG.info("Starting the cloning of SameDiff variables from a loaded model");
            // Cloning all relevant variables (weights, biases etc.)
            var variableTypes = Set.of(VARIABLE, CONSTANT);
            loadedSameDiff.getVariables().values().stream()
                    .map(Variable::getVariable)
                    .filter(variable -> variableTypes.contains(variable.getVariableType()))
                    .filter(variable -> !variable.name().endsWith("-grad"))
                    .forEach(this::copyVariableData);
            LOG.info("Completed the cloning of SameDiff variables");
        } catch (ObjectStreamException e) {
            // Something went wrong during SameDiff load - could be a different version or something else, so just logging that info
            LOG.warn("SameDiff couldn't be deserialized from zip file entry {}", sameDiffFile);
        }
    }

    private void copyVariableData(SDVariable sourceVariable) {
        List<String> correspondingActualVariableNames = this.sameDiff.getVariables().keySet().stream()
                .filter(sourceVariable.name()::equals)
                .toList();
        correspondingActualVariableNames.forEach(targetVariableName ->
                copyVariableArray(sourceVariable, targetVariableName, "Copied variable data"));
    }

    private void copyVariableArray(SDVariable sourceVariable, String targetVariableName, String copyDescription) {
        var targetShape = this.sameDiff.getVariable(targetVariableName).getShape();
        var sourceShape = sourceVariable.getArr().shape();
        if (Arrays.equals(targetShape, sourceShape)) {
            this.sameDiff.setArrayForVariable(targetVariableName, sourceVariable.getArr());
            LOG.debug("{} from '{}' into '{}'", copyDescription, sourceVariable.name(), targetVariableName);
        } else {
            LOG.warn("Ignored copying data for variable '{}' because of different array shapes: source <{}> vs. target <{}> ",
                    sourceVariable.name(), Arrays.toString(sourceShape), Arrays.toString(targetShape));
        }
    }

    public static abstract class Builder<T extends Builder<T, U>, U extends AbstractTransformerSameDiffModel<U>> {
        protected int batchSize = 500;
        protected int intermediateLayerSize = 1024;
        protected double learningRate = 3e-5;
        protected int layersAmount = 4;
        protected double labelSmoothing = 0;
        protected int loggingFrequency = 50;
        protected int attentionHeadsAmount = 8;
        protected int modelTestFrequency = 100;
        protected ISchedule optimizerSchedule;
        protected int sequenceLength = 256;
        protected double beta2 = 0.98;
        protected int hiddenSize = 768;
        protected int saveFrequencyInSteps = 100;
        protected final WordPieceVocab vocab;
        protected boolean fixWeights;
        protected boolean fixTokenEmbeddings;
        protected float dropout;

        public Builder(WordPieceVocab vocab) {
            this.vocab = requireNonNull(vocab);
        }

        protected abstract T getInstance();

        public T withSequenceLength(int sequenceLength) {
            this.sequenceLength = sequenceLength;
            return getInstance();
        }

        public T withSaveFrequencyInSteps(int saveFrequencyInSteps) {
            this.saveFrequencyInSteps = saveFrequencyInSteps;
            return getInstance();
        }

        public T withDropout(float dropout) {
            this.dropout = dropout;
            return getInstance();
        }

        public T withBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return getInstance();
        }

        public T withBeta2(double beta2) {
            this.beta2 = beta2;
            return getInstance();
        }

        public T withHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return getInstance();
        }

        public T withModelTestFrequency(int modelTestFrequency) {
            this.modelTestFrequency = modelTestFrequency;
            return getInstance();
        }

        public T withLoggingFrequency(int loggingFrequency) {
            this.loggingFrequency = loggingFrequency;
            return getInstance();
        }

        public T withLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return getInstance();
        }

        public T withOwnLayersAmount(int layersAmount) {
            this.layersAmount = layersAmount;
            return getInstance();
        }

        public T withAttentionHeadsAmount(int attentionHeadsAmount) {
            this.attentionHeadsAmount = attentionHeadsAmount;
            return getInstance();
        }

        public T withIntermediateLayerSize(int intermediateLayerSize) {
            this.intermediateLayerSize = intermediateLayerSize;
            return getInstance();
        }

        public <V extends ISchedule> T withOptimizerSchedule(V optimizerSchedule) {
            this.optimizerSchedule = optimizerSchedule;
            return getInstance();
        }

        public T withLabelSmoothing(double labelSmoothing) {
            this.labelSmoothing = labelSmoothing;
            return getInstance();
        }

        public T withFixedWeights(boolean fixed) {
            this.fixWeights = fixed;
            return getInstance();
        }

        public T withFixedTokenEmbeddings(boolean fixed) {
            this.fixTokenEmbeddings = fixed;
            return getInstance();
        }

        public abstract U build();
    }
}