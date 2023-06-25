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

package org.tarik.core.network.layers.sd.transformer.attention;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.List;
import java.util.function.Function;

import static java.util.Objects.requireNonNull;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;
import static org.tarik.utils.SdUtils.*;

/**
 * An abstract class which contains the common logic of the Scaled Dot-Product Attention Mechanism based on the
 * <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>. It allows to add create the corresponding sub-graph to the specified
 * instance of {@link SameDiff}. All Samediff variables are created in a separate name scope. It also allows to freeze (turn into constants)
 * all the weights (could be useful if existing model's weights are copied into the current Samediff instance and this layer needs to be
 * frozen). If the dropout rate > 0, it will be applied to the attention output.
 */
public abstract class AttentionLayerGraphGenerator {
    public static final String ATTENTION_SCORES_VAR_NAME = "attentionSoftmaxScores";
    public static final String ATTENTION_WEIGHT_VALUES_VAR_NAME = "attentionWeightsMasked";
    public static final String KEY_WEIGHTS_VAR_NAME = "AttentionKeyWeights";
    public static final String QUERY_WEIGHTS_VAR_NAME = "AttentionQueryWeights";
    public static final String VALUE_WEIGHTS_VAR_NAME = "AttentionValueWeights";
    public static final String ATTENTION_OUT_LAYER_WEIGHTS_VAR_NAME = "AttentionOutWeights";
    public static final String ATTENTION_OUTPUT_VAR_NAME = "attentionOutput";

    protected final SameDiff sameDiff;
    protected final int attentionHeadsAmount;
    protected final int attentionHeadEmbeddingSize;
    protected final int hiddenSize;
    protected final boolean freezeAllWeights;
    protected final double dropoutRate;
    protected final SDVariable attentionDotProductShape;
    protected final double attentionScoresScalingFactor;
    protected final SDVariable attentionHeadsAmountSd;
    protected final SDVariable attentionHeadEmbeddingSizeSd;
    protected final SDVariable batchSize;
    protected final SDVariable queriesSequenceLength;
    protected final SDVariable keysAndValuesPerHeadAttentionShape;
    protected final SDVariable queriesPerHeadAttentionShape;
    protected final String id;

    public AttentionLayerGraphGenerator(SameDiff sameDiff, int attentionHeadsAmount, int attentionHeadEmbeddingSize,
                                        int hiddenSize, SDVariable batchSize, SDVariable queriesSequenceLength,
                                        SDVariable keysAndValuesSequenceLength, boolean freezeAllWeights, double dropoutRate, String id) {
        this.id = id;
        try (var ignored = sameDiff.withNameScope(String.format("%s_%s", getAttentionType(), id))) {
            this.sameDiff = sameDiff;
            this.attentionHeadsAmount = attentionHeadsAmount;
            this.attentionHeadEmbeddingSize = attentionHeadEmbeddingSize;
            this.hiddenSize = hiddenSize;
            this.freezeAllWeights = freezeAllWeights;
            this.dropoutRate = dropoutRate;
            this.attentionHeadsAmountSd = getOrCreateConstant(sameDiff, "attentionHeadsAmount", attentionHeadsAmount);
            this.attentionHeadEmbeddingSizeSd = getOrCreateConstant(sameDiff, "attentionHeadEmbeddingSize", attentionHeadEmbeddingSize);
            this.attentionDotProductShape = this.sameDiff.stack("attentionDotProductShape", 0, batchSize.mul(queriesSequenceLength),
                    attentionHeadsAmountSd.mul(attentionHeadEmbeddingSizeSd));
            this.batchSize = batchSize;
            this.attentionScoresScalingFactor = 1.0 / Math.sqrt(attentionHeadEmbeddingSize);
            this.queriesSequenceLength = queriesSequenceLength;
            this.keysAndValuesPerHeadAttentionShape = sameDiff.stack("keysPerHeadAttentionShape", 0, batchSize,
                    keysAndValuesSequenceLength, attentionHeadsAmountSd, attentionHeadEmbeddingSizeSd);
            this.queriesPerHeadAttentionShape = sameDiff.stack("queriesPerHeadAttentionShape", 0, batchSize, queriesSequenceLength,
                    attentionHeadsAmountSd, attentionHeadEmbeddingSizeSd);
        }
    }

    protected abstract String getAttentionType();

    protected String getSuffix(int layerIndex) {
        return "_" + layerIndex;
    }

    public AttentionResult generateGraph(int layerIndex, SDVariable keys, SDVariable values, SDVariable queries,
                                         SDVariable attentionMasks) {
        requireNonNull(keys, "Input for Attention Keys must be present");
        requireNonNull(values, "Input for Attention Values must be present");
        requireNonNull(queries, "Input for Attention Queries must be present");

        Function<String, SDVariable> attentionWeightsProvider =
                name -> sameDiff.var(name, new XavierInitScheme('c', attentionHeadsAmount * attentionHeadEmbeddingSize, hiddenSize),
                        FLOAT, hiddenSize, (long) attentionHeadsAmount * attentionHeadEmbeddingSize);
        try (var ignored = sameDiff.withNameScope(String.format("%s_%s", getAttentionType(), id))) {
            var queryWeights = getOrCreateVariable(sameDiff, QUERY_WEIGHTS_VAR_NAME + getSuffix(layerIndex), attentionWeightsProvider);
            var valueWeights = getOrCreateVariable(sameDiff, VALUE_WEIGHTS_VAR_NAME + getSuffix(layerIndex), attentionWeightsProvider);
            var keyWeights = getOrCreateVariable(sameDiff, KEY_WEIGHTS_VAR_NAME + getSuffix(layerIndex), attentionWeightsProvider);

            var attentionOutWeights = getOrCreateVariable(sameDiff, ATTENTION_OUT_LAYER_WEIGHTS_VAR_NAME + getSuffix(layerIndex),
                    name -> sameDiff.var(name, new XavierInitScheme('c', attentionHeadsAmount * attentionHeadEmbeddingSize, hiddenSize),
                            FLOAT, (long) attentionHeadsAmount * attentionHeadEmbeddingSize, hiddenSize));

            if (freezeAllWeights) {
                sameDiff.convertToConstants(List.of(queryWeights, valueWeights, keyWeights, attentionOutWeights));
            }

            var attentionMaskDisqualifier = attentionMasks.sub(1)
                    .mul(getUniqueName("attentionMaskDisqualifier" + getSuffix(layerIndex)), 1e10);

            var keyProjections = keys.mmul(getUniqueName("keyProjections" + getSuffix(layerIndex)), keyWeights);
            var queryProjections = queries.mmul(getUniqueName("queryProjections" + getSuffix(layerIndex)), queryWeights);
            var valueProjections = values.mmul(getUniqueName("valueProjections" + getSuffix(layerIndex)), valueWeights);

            //[Batch size, att. heads amount, seq. length, head_size]
            var keysReshapedForScores = reshapeAndTransposeForAttentionScores(keyProjections, keysAndValuesPerHeadAttentionShape);
            var queriesReshapedForScores = reshapeAndTransposeForAttentionScores(queryProjections, queriesPerHeadAttentionShape);
            var valuesReshapedForOutput = reshapeAndTransposeForAttentionScores(valueProjections, keysAndValuesPerHeadAttentionShape);

            //[Batch size, att. heads amount, queries seq. length, keys seq. length]
            var attentionScoresBeforeMasking = sameDiff.mmul(queriesReshapedForScores, keysReshapedForScores, false, true, false)
                    .mul(getUniqueName("attentionScoresBeforeMasking" + getSuffix(layerIndex)), attentionScoresScalingFactor);
            var attentionScoresMasked = attentionScoresBeforeMasking.add(
                    getUniqueName(ATTENTION_WEIGHT_VALUES_VAR_NAME + getSuffix(layerIndex)), attentionMaskDisqualifier);
            var attentionSoftmaxScores = sameDiff.nn().softmax(getUniqueName(ATTENTION_SCORES_VAR_NAME + getSuffix(layerIndex)),
                    attentionScoresMasked, 3);

            //[Batch size, att. heads amount, queries seq. length, head_size]
            var valuesBasedOnAttentionScores =
                    attentionSoftmaxScores.mmul(getUniqueName("valuesBasedOnAttentionScores" + getSuffix(layerIndex)), valuesReshapedForOutput);

            //[Batch size*queries seq. length, hidden_size]
            var attentionDotProductOutput = valuesBasedOnAttentionScores.permute(0, 2, 1, 3).reshape(attentionDotProductShape)
                    .rename(getUniqueName("attentionDotProductOutput" + getSuffix(layerIndex)));
            var attentionOutput = attentionDotProductOutput.mmul(attentionOutWeights);

            if (dropoutRate > 0) {
                attentionOutput = sameDiff.nn().dropout(attentionOutput, false, 1 - dropoutRate);
            }

            sameDiff.updateVariableNameAndReference(attentionOutput, getUniqueName(ATTENTION_OUTPUT_VAR_NAME + getSuffix(layerIndex)));

            return new AttentionResult(attentionOutput, attentionSoftmaxScores, attentionScoresMasked);
        }
    }

    protected String getUniqueName(String namePrefix) {
        return generateUniqueVariableName(sameDiff, namePrefix);
    }

    private SDVariable reshapeAndTransposeForAttentionScores(SDVariable batchedInput, SDVariable perHeadShape) {
        var inputReshapedPerHead = batchedInput.reshape(perHeadShape);

        //[Batch size, att. heads amount, seq. length, head_size]
        return inputReshapedPerHead.permute(0, 2, 1, 3);
    }

    public record AttentionResult(SDVariable attentionOutput, SDVariable attentionSoftmaxScores,
                                  SDVariable attentionWeightValuesBeforeSoftmax) {
    }
}