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

package org.tarik.core.network.layers.sd.transformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.tarik.core.network.layers.sd.transformer.attention.CrossAttentionLayerGraphGenerator;
import org.tarik.core.network.layers.sd.transformer.attention.SelfAttentionLayerGraphGenerator;
import org.tarik.utils.SdUtils;

import javax.annotation.Nonnull;

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Objects.requireNonNull;
import static org.tarik.utils.SdUtils.*;

/**
 * An abstract class representing the common functionality of Encoder and Decoder Stacks (Blocks)
 * based on the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>.
 * It's normally responsible for providing the attention and hidden layer graph generators as well as attention masks.
 */
public abstract class TransformerBlockGraphGenerator {
    protected final int sequenceLength;
    protected final int hiddenSize;
    protected final int intermediateLayerSize;
    protected final int layersAmount;
    protected final int attentionHeadsAmount;
    protected final boolean fixWeights;
    protected final int attentionHeadEmbeddingSize;
    protected final double dropoutRate;
    protected final String id;

    public TransformerBlockGraphGenerator(int sequenceLength, int hiddenSize, int intermediateLayerSize, int layersAmount,
                                          int attentionHeadsAmount, boolean fixWeights, double dropoutRate, String id) {
        checkArgument(layersAmount > 0, "TransformerBlockGraphGenerator must have at least 1 layer");
        this.sequenceLength = sequenceLength;
        this.hiddenSize = hiddenSize;
        this.intermediateLayerSize = intermediateLayerSize;
        this.layersAmount = layersAmount;
        this.attentionHeadsAmount = attentionHeadsAmount;
        this.fixWeights = fixWeights;
        this.dropoutRate = dropoutRate;
        this.id = id;
        this.attentionHeadEmbeddingSize = hiddenSize / attentionHeadsAmount;
    }

    protected SDVariable createCrossAttentionMasks(SameDiff sd, SDVariable batchQueriesInputMask, SDVariable batchKeysInputMask,
                                                   boolean clsTokenPresent, String id) {
        try (var ignored = sd.withNameScope(id)) {
            var actualQueriesMask = batchQueriesInputMask;
            if (clsTokenPresent) {
                int targetDimension = 1;
                var queriesSequenceLength = getDimensionSize(batchQueriesInputMask, targetDimension);
                var queryTokenPositions = getPositionsRange(sd, getOrCreateConstant(sd, "one", 1), queriesSequenceLength);
                var newClsTokenMasks = gather(sd, sd.onesLike(batchQueriesInputMask), 0, targetDimension);
                var extendedMasksWithoutCls = gather(sd, batchQueriesInputMask, queryTokenPositions, targetDimension);
                actualQueriesMask = sd.concat(targetDimension, newClsTokenMasks, extendedMasksWithoutCls);
            }

            //[Batch size, 1, keys seq. length]
            var keysMaskExpanded = sd.expandDims("keysMaskExpanded", batchKeysInputMask, 1);
            //[Batch size, queries. length, 1]
            var queriesMaskExpanded = sd.expandDims("queriesMaskExpanded", actualQueriesMask, 2);
            // [Batch size, queries seq. length, keys seq. length]
            var combinedMask = queriesMaskExpanded.mul("combinedMask", keysMaskExpanded);
            // [Batch size, 1, queries seq. length, keys seq. length]
            return sd.expandDims("crossAttentionMasks", combinedMask, 1);
        }
    }

    protected SDVariable createCrossAttentionMasks(SameDiff sd, SDVariable batchQueriesInputMask, SDVariable batchKeysInputMask,
                                                   String id) {
        return createCrossAttentionMasks(sd, batchQueriesInputMask, batchKeysInputMask, true, id);
    }

    protected SDVariable createSelfAttentionMasks(SameDiff sd, SDVariable batchTokenInputMask) {
        var maskBatchSize = getDimensionSize(batchTokenInputMask, 0);
        var maskSequenceLength = getDimensionSize(batchTokenInputMask, 1);
        var attentionIntermediateMaskShape =
                sd.stack("attentionIntermediateMaskShape", 0, maskBatchSize, sd.constant(1), maskSequenceLength);
        // [Batch size, 1, seq. length]
        var attentionIntermediateMask = batchTokenInputMask.reshape(attentionIntermediateMaskShape)
                .rename(getUniqueName(sd, "attentionIntermediateMask_"));
        // [Batch size, seq. length, 1]
        var broadcastOnes = sd.expandDims(sd.onesLike("broadcastOnes_" + id, batchTokenInputMask), 2);
        // [Batch size, seq. length, seq. length]
        var intermediateSelfAttentionMasks = broadcastOnes.mul("intermediateSelfAttentionMasks_" + id,
                attentionIntermediateMask);
        // [Batch size, 1, seq. length, seq. length]
        return sd.expandDims("selfAttentionMasks", intermediateSelfAttentionMasks, 1);
    }

    private String getUniqueName(SameDiff sd, String base) {
        return SdUtils.generateUniqueVariableName(sd, base);
    }

    protected SDVariable createEncoderDecoderCrossAttentionMasks(SameDiff sd, SDVariable batchDecoderSelfAttentionCausalMask,
                                                                 SDVariable batchEncoderSelfAttentionMask, String id) {
        // last sequence element should have the attention which corresponds to the self-attention
        var batchDecoderSelfAttentionMask = gather(sd, "batchDecoderSelfAttentionMask",
                sd.squeeze(batchDecoderSelfAttentionCausalMask, 1), sequenceLength - 1, 1);
        sd.squeeze(batchDecoderSelfAttentionMask, 1);
        return createCrossAttentionMasks(sd, sd.squeeze(batchDecoderSelfAttentionMask, 1), batchEncoderSelfAttentionMask, false, id);
    }

    protected SDVariable createSelfAttentionCausalMasks(SameDiff sd, SDVariable batchTokenInputMasks, SDVariable causalMasksTemplate) {
        // [1, 1, seq. length, seq. length]
        var causalAttentionMasksExpanded = sd.expandDims("causalAttentionMasksExpanded_" + id,
                sd.expandDims(causalMasksTemplate, 0), 0);
        // [Batch size, 1, 1, seq. length]
        var batchInputMasksExpanded = sd.expandDims("batchInputMasksExpanded_" + id,
                sd.expandDims(batchTokenInputMasks, 1), 1);
        // [Batch size, 1, seq. length, seq. length]
        return batchInputMasksExpanded.mul("selfAttentionCausalMasks_" + id, causalAttentionMasksExpanded);
    }

    @Nonnull
    protected HiddenLayerBlockGraphGenerator getHiddenLayerBlockGraphGenerator(SameDiff sd, String idSuffix) {
        return new HiddenLayerBlockGraphGenerator.Builder(sd)
                .withIntermediateLayerSize(intermediateLayerSize)
                .withHiddenSize(hiddenSize)
                .withFixedWeights(fixWeights)
                .withDropoutRate(dropoutRate)
                .withIdSuffix(idSuffix)
                .build();
    }

    @Nonnull
    protected HiddenLayerBlockGraphGenerator getHiddenLayerBlockGraphGenerator(SameDiff sd) {
        return getHiddenLayerBlockGraphGenerator(sd, null);
    }

    @Nonnull
    protected CrossAttentionLayerGraphGenerator getCrossAttentionLayerGraphGenerator(SameDiff sd,
                                                                                     int attentionHeadEmbeddingSize,
                                                                                     SDVariable sequenceLengthSd, SDVariable batchSize,
                                                                                     SDVariable queriesPositionalEmbeddingsForAttention,
                                                                                     SDVariable keysPositionalEmbeddingsForAttention,
                                                                                     SDVariable externalSequenceLength,
                                                                                     boolean withFixedWeights, String id) {
        return new CrossAttentionLayerGraphGenerator.Builder(sd, id)
                .withBatchSize(batchSize)
                .withHiddenSize(hiddenSize)
                .withSequenceLength(sequenceLengthSd)
                .withAttentionHeadEmbeddingSize(attentionHeadEmbeddingSize)
                .withAttentionHeadsAmount(attentionHeadsAmount)
                .withKeysAndValuesSequenceLength(externalSequenceLength)
                .withQueriesPositionalEmbeddingsForAttention(queriesPositionalEmbeddingsForAttention)
                .withKeysPositionalEmbeddingsForAttention(keysPositionalEmbeddingsForAttention)
                .withFixedWeights(withFixedWeights)
                .withDropoutRate(dropoutRate)
                .build();
    }

    @Nonnull
    protected SelfAttentionLayerGraphGenerator getSelfAttentionLayerGraphGenerator(SameDiff sd, int attentionHeadEmbeddingSize,
                                                                                   SDVariable sequenceLengthSd, SDVariable batchSize,
                                                                                   SDVariable positionalEmbeddingsForAttention, String id) {
        return new SelfAttentionLayerGraphGenerator.Builder(sd, id)
                .withBatchSize(batchSize)
                .withHiddenSize(hiddenSize)
                .withSequenceLength(sequenceLengthSd)
                .withAttentionHeadEmbeddingSize(attentionHeadEmbeddingSize)
                .withAttentionHeadsAmount(attentionHeadsAmount)
                .withPositionalEmbeddingsForAttention(positionalEmbeddingsForAttention)
                .withFixedWeights(fixWeights)
                .withDropoutRate(dropoutRate)
                .build();
    }

    public static abstract class Builder<T extends Builder<T, U>, U extends TransformerBlockGraphGenerator> {
        protected int sequenceLength;
        protected int hiddenSize;
        protected int intermediateLayerSize;
        protected int layersAmount;
        protected int attentionHeadsAmount;
        protected boolean fixWeights;
        protected int attentionHeadEmbeddingSize;
        protected double dropoutRate;
        protected final String id;

        public Builder(String id) {
            this.id = requireNonNull(id);
        }

        protected abstract T getInstance();

        public T withSequenceLength(int sequenceLength) {
            this.sequenceLength = sequenceLength;
            return getInstance();
        }

        public T withHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return getInstance();
        }

        public T withIntermediateLayerSize(int intermediateLayerSize) {
            this.intermediateLayerSize = intermediateLayerSize;
            return getInstance();
        }

        public T withLayersAmount(int layersAmount) {
            this.layersAmount = layersAmount;
            return getInstance();
        }

        public T withAttentionHeadsAmount(int attentionHeadsAmount) {
            this.attentionHeadsAmount = attentionHeadsAmount;
            return getInstance();
        }

        public T withFixWeights(boolean fixWeights) {
            this.fixWeights = fixWeights;
            return getInstance();
        }

        public T withAttentionHeadEmbeddingSize(int attentionHeadEmbeddingSize) {
            this.attentionHeadEmbeddingSize = attentionHeadEmbeddingSize;
            return getInstance();
        }

        public T withDropoutRate(double dropoutRate) {
            this.dropoutRate = dropoutRate;
            return getInstance();
        }

        public abstract U build();
    }
}