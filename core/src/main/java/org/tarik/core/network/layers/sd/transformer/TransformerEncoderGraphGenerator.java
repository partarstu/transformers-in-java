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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import static java.lang.String.format;
import static java.util.Optional.ofNullable;
import static org.tarik.utils.SdUtils.*;

/**
 * A separate Encoder Block (Stack) based on the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>.
 * Positional embeddings are optional. Traditional self-attention masks are used and are generated based on the specified simple token
 * presence masks for each sequence.
 */
public class TransformerEncoderGraphGenerator extends TransformerBlockGraphGenerator {
    public static final String ENCODER_OUTPUT_VAR_NAME = "encoderOutputNormalized";

    private static String getUniqueId(String encoderId) {
        return encoderId;
    }

    public static String getEncoderOutputVarName(String encoderId) {
        return format("%s/%s", getUniqueId(encoderId), ENCODER_OUTPUT_VAR_NAME);
    }

    public TransformerEncoderGraphGenerator(String id, int sequenceLength, int hiddenSize, int intermediateLayerSize,
                                            int encoderLayersAmount, int attentionHeadsAmount, boolean fixEncoderWeights,
                                            double dropoutRate) {
        super(sequenceLength, hiddenSize, intermediateLayerSize, encoderLayersAmount, attentionHeadsAmount, fixEncoderWeights, dropoutRate,
                id);
    }

    public SDVariable generateGraph(@Nonnull SameDiff sd, @Nonnull SDVariable inputTokenMasks, @Nonnull SDVariable batchTokenEmbeddings,
                                    @Nullable SDVariable positionalEmbeddings) {
        SDVariable lastLayerOutput = null;
        boolean firstLayerInitialized = false;

        try (var ignored = sd.withNameScope(id)) {
            var sequenceLengthSd = getOrCreateConstant(sd, "encSequenceLength", sequenceLength);
            var batchSize = getDimensionSize("encBatchSize", batchTokenEmbeddings, 0);
            var flatBatchSizeSd = batchSize.mul(sequenceLengthSd);
            var hiddenSizeSd = getOrCreateConstant(sd, "encHiddenSize", hiddenSize);
            var hiddenLayerInputShape = sd.stack("encHiddenLayerInputShape", 0, flatBatchSizeSd, hiddenSizeSd);
            var one = getOrCreateConstant(sd, "one", 1);
            var attentionMasks = createSelfAttentionMasks(sd, inputTokenMasks);
            var positionalEmbeddingsForAttention = ofNullable(positionalEmbeddings)
                    .map(embeddings -> sd.tile("positionalEmbeddingsForAttention", positionalEmbeddings, sd.stack(0, batchSize, one)));

            var selfAttentionLayerGraphGenerator = getSelfAttentionLayerGraphGenerator(sd, attentionHeadEmbeddingSize,
                    sequenceLengthSd, batchSize, positionalEmbeddingsForAttention.orElse(null), "default");
            var hiddenLayerBlockGraphGenerator = getHiddenLayerBlockGraphGenerator(sd);

            for (int i = 0; i < layersAmount; i++) {
                // Input for self-attention - either initial token embeddings, or the output of the last layer
                var layerInput = firstLayerInitialized ? lastLayerOutput : batchTokenEmbeddings.reshape(hiddenLayerInputShape);
                var selfAttentionResult = selfAttentionLayerGraphGenerator.generateGraph(i, layerInput, attentionMasks);
                lastLayerOutput = hiddenLayerBlockGraphGenerator.generateGraph(i,
                        selfAttentionResult.selfAttentionResidualProduct()).layerResidualProduct();
                firstLayerInitialized = true;
            }

            return normalizeLayerWithNoGain(sd, ENCODER_OUTPUT_VAR_NAME, lastLayerOutput, 1, hiddenSize);
        }
    }

    public static final class Builder extends TransformerBlockGraphGenerator.Builder<Builder, TransformerEncoderGraphGenerator> {

        public Builder(String id) {
            super(id);
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public TransformerEncoderGraphGenerator build() {
            return new TransformerEncoderGraphGenerator(id, sequenceLength, hiddenSize, intermediateLayerSize, layersAmount,
                    attentionHeadsAmount, fixWeights, dropoutRate);
        }
    }
}