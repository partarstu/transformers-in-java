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
 * Scaled Dot-Product Self-attention based on the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>
 * with some modifications based on <a href="https://arxiv.org/pdf/2012.15832.pdf">Position-Infused behavior</a>.
 * </p>
 * <p>
 * This layer presumes that the queries, keys and values are the same. Layer normalization is applied to all inputs. Positional
 * embeddings are expected only for queries and keys (based on the mentioned above paper it provides better results). Residual
 * connections are applied based on the original input before layer normalization.
 * </p>
 */
public class SelfAttentionLayerGraphGenerator extends AttentionLayerGraphGenerator {

    protected final SDVariable positionalEmbeddingsForAttention;

    protected SelfAttentionLayerGraphGenerator(SameDiff sameDiff, int attentionHeadsAmount, int attentionHeadEmbeddingSize,
                                               int hiddenSize, SDVariable batchSize, SDVariable sequenceLengthSd, boolean freezeAllWeights,
                                               SDVariable positionalEmbeddingsForAttention, double dropoutRate, String id) {
        super(sameDiff, attentionHeadsAmount, attentionHeadEmbeddingSize, hiddenSize, batchSize, sequenceLengthSd, sequenceLengthSd,
                freezeAllWeights, dropoutRate, id);
        this.positionalEmbeddingsForAttention = positionalEmbeddingsForAttention;
    }

    public SelfAttentionResult generateGraph(int layerIndex, SDVariable attentionInput, SDVariable attentionMasks) {
        try (var ignored = sameDiff.withNameScope(String.valueOf(layerIndex))) {
            var normalizedInput = normalizeLayerWithNoGain(sameDiff, "normalizedAttentionInput", attentionInput, 1, hiddenSize);

            SDVariable keyAndQueryInput = normalizedInput;
            if (positionalEmbeddingsForAttention != null) {
                keyAndQueryInput = sameDiff.math.add("keyAndQueryInput_WithPositionalEmbed", attentionInput,
                        positionalEmbeddingsForAttention);
                keyAndQueryInput = normalizeLayerWithNoGain(sameDiff, "keyAndQueryInputNormalized", keyAndQueryInput, 1, hiddenSize);
            }

            var attentionResult = super.generateGraph(layerIndex, keyAndQueryInput, normalizedInput, keyAndQueryInput, attentionMasks);
            var selfAttentionResidualProduct = attentionInput.add("selfAttentionResidualProduct", attentionResult.attentionOutput());

            return new SelfAttentionResult(attentionResult, selfAttentionResidualProduct);
        }
    }

    @Override
    protected String getAttentionType() {
        return "self_attention";
    }

    public record SelfAttentionResult(AttentionResult attentionResult, SDVariable selfAttentionResidualProduct) {
    }

    public static final class Builder {
        private final SameDiff sameDiff;
        private int hiddenSize = 768;
        private int attentionHeadsAmount = 8;
        private int attentionHeadEmbeddingSize = hiddenSize / attentionHeadsAmount;
        private boolean fixEncoderWeights;
        private double dropoutRate = 0;
        private SDVariable positionalEmbeddingsForAttention;
        private SDVariable batchSize;
        private SDVariable sequenceLength;
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

        public Builder withPositionalEmbeddingsForAttention(SDVariable positionalEmbeddingsForAttention) {
            this.positionalEmbeddingsForAttention = positionalEmbeddingsForAttention;
            return this;
        }

        public Builder withBatchSize(SDVariable batchSize) {
            this.batchSize = requireNonNull(batchSize);
            return this;
        }

        public Builder withSequenceLength(SDVariable sequenceLength) {
            this.sequenceLength = requireNonNull(sequenceLength);
            return this;
        }

        public SelfAttentionLayerGraphGenerator build() {
            return new SelfAttentionLayerGraphGenerator(sameDiff, attentionHeadsAmount, attentionHeadEmbeddingSize, hiddenSize,
                    batchSize, sequenceLength, fixEncoderWeights, positionalEmbeddingsForAttention, dropoutRate, id);
        }
    }
}