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

import static java.util.Objects.requireNonNull;
import static org.tarik.utils.SdUtils.normalizeLayerWithNoGain;

/**
 * <p>
 * Scaled Dot-Product Cross-attention based on the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>
 * with some modifications based on <a href="https://arxiv.org/pdf/2012.15832.pdf">Position-Infused behavior</a>.
 * </p>
 * <p>
 * This layer presumes that the queries are different from the keys and values. Layer normalization is applied to all inputs. Positional
 * embeddings are expected only for queries and keys (based on the mentioned above paper it provides better results). Residual
 * connections are applied based on the original value of queries before layer normalization.
 * </p>
 */
public class CrossAttentionLayerGraphGenerator extends AttentionLayerGraphGenerator {

    protected final SDVariable queriesPositionalEmbeddingsForAttention;
    protected final SDVariable keysPositionalEmbeddingsForAttention;

    protected CrossAttentionLayerGraphGenerator(SameDiff sameDiff, int attentionHeadsAmount, int attentionHeadEmbeddingSize,
                                                int hiddenSize, SDVariable batchSize,
                                                SDVariable sequenceLengthSd, SDVariable keysAndValuesSequenceLength,
                                                boolean freezeAllWeights,
                                                SDVariable queriesPositionalEmbeddingsForAttention, double dropoutRate,
                                                SDVariable keysPositionalEmbeddingsForAttention, String id) {
        super(sameDiff, attentionHeadsAmount, attentionHeadEmbeddingSize, hiddenSize, batchSize,
                sequenceLengthSd, keysAndValuesSequenceLength, freezeAllWeights, dropoutRate, id);
        this.queriesPositionalEmbeddingsForAttention = queriesPositionalEmbeddingsForAttention;
        this.keysPositionalEmbeddingsForAttention = keysPositionalEmbeddingsForAttention;
    }

    public CrossAttentionResult generateGraph(int layerIndex, SDVariable queries, SDVariable keysAndValues, SDVariable attentionMasks) {
        var queriesInput = normalizeLayerWithNoGain(sameDiff, getUniqueName("queriesNormalized"), queries, 1, hiddenSize);
        var valuesInput = normalizeLayerWithNoGain(sameDiff, getUniqueName("keysNormalized"), keysAndValues, 1, hiddenSize);
        SDVariable keysInput = valuesInput;

        try (var ignored = sameDiff.withNameScope(String.format("%s_%s", id, layerIndex))) {
            if (queriesPositionalEmbeddingsForAttention != null) {
                queriesInput = sameDiff.math.add(getUniqueName("queriesWithPositionalInfo"), queries,
                        queriesPositionalEmbeddingsForAttention);
                queriesInput = normalizeLayerWithNoGain(sameDiff, getUniqueName("queriesWithPositionalInfoNormalized"), queriesInput, 1,
                        hiddenSize);
            }

            if (keysPositionalEmbeddingsForAttention != null) {
                keysInput = sameDiff.math.add(getUniqueName("keysWithPositionalInfo"), keysInput, keysPositionalEmbeddingsForAttention);
                keysInput = normalizeLayerWithNoGain(sameDiff, getUniqueName("keysWithPositionalInfoNormalized"), keysInput, 1, hiddenSize);
            }

            var attentionResult = super.generateGraph(layerIndex, keysInput, valuesInput, queriesInput, attentionMasks);
            var crossAttentionResidualProduct = queries.add("crossAttentionResidualProduct", attentionResult.attentionOutput());

            return new CrossAttentionResult(attentionResult, crossAttentionResidualProduct);
        }
    }

    @Override
    protected String getAttentionType() {
        return "cross_attention";
    }

    public record CrossAttentionResult(AttentionResult attentionResult, SDVariable residualProduct) {
    }

    public static final class Builder {
        private final SameDiff sameDiff;
        private int hiddenSize = 768;
        private int attentionHeadsAmount = 8;
        private int attentionHeadEmbeddingSize = hiddenSize / attentionHeadsAmount;
        private boolean fixEncoderWeights;
        private double dropoutRate = 0;
        private SDVariable queriesPositionalEmbeddingsForAttention;
        private SDVariable keysPositionalEmbeddingsForAttention;
        private SDVariable batchSize;
        private SDVariable sequenceLength;
        private SDVariable keysAndValuesSequenceLength;
        private final String id;

        public Builder(SameDiff sameDiff, String id) {
            this.sameDiff = requireNonNull(sameDiff);
            this.id = requireNonNull(id);
        }

        public Builder withDropoutRate(double dropoutRate) {
            this.dropoutRate = dropoutRate;
            return this;
        }

        public Builder withAttentionHeadsAmount(int attentionHeadsAmount) {
            this.attentionHeadsAmount = attentionHeadsAmount;
            return this;
        }

        public Builder withAttentionHeadEmbeddingSize(int attentionHeadEmbeddingSize) {
            this.attentionHeadEmbeddingSize = attentionHeadEmbeddingSize;
            return this;
        }

        public Builder withHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder withFixedWeights(boolean fixEncoderWeights) {
            this.fixEncoderWeights = fixEncoderWeights;
            return this;
        }

        public Builder withQueriesPositionalEmbeddingsForAttention(SDVariable positionalEmbeddingsForAttention) {
            this.queriesPositionalEmbeddingsForAttention = positionalEmbeddingsForAttention;
            return this;
        }

        public Builder withKeysPositionalEmbeddingsForAttention(SDVariable positionalEmbeddingsForAttention) {
            this.keysPositionalEmbeddingsForAttention = positionalEmbeddingsForAttention;
            return this;
        }

        public Builder withBatchSize(SDVariable batchSize) {
            this.batchSize = requireNonNull(batchSize);
            return this;
        }

        public Builder withKeysAndValuesSequenceLength(SDVariable keysAndValuesSequenceLength) {
            this.keysAndValuesSequenceLength = requireNonNull(keysAndValuesSequenceLength);
            return this;
        }

        public Builder withSequenceLength(SDVariable sequenceLength) {
            this.sequenceLength = requireNonNull(sequenceLength);
            return this;
        }

        public CrossAttentionLayerGraphGenerator build() {
            return new CrossAttentionLayerGraphGenerator(sameDiff, attentionHeadsAmount, attentionHeadEmbeddingSize,
                    hiddenSize, batchSize, sequenceLength, keysAndValuesSequenceLength, fixEncoderWeights,
                    queriesPositionalEmbeddingsForAttention, dropoutRate, keysPositionalEmbeddingsForAttention, id);
        }
    }
}