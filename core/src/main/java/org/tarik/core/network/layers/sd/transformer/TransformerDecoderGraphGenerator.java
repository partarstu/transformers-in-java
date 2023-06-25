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

import static java.util.Optional.ofNullable;
import static org.tarik.utils.SdUtils.getDimensionSize;
import static org.tarik.utils.SdUtils.normalizeLayerWithNoGain;

/**
 * A separate generational Decoder Block (Stack) based on the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>.
 * Cross-attention is not used here because the encoder part is missing. Positional embeddings are optional. Instead of traditional
 * self-attention masks the so-called "causal" masks (the ones masking for each token all other tokens which are positioned after it in
 * the sequence) are used.
 */
public class TransformerDecoderGraphGenerator extends TransformerBlockGraphGenerator {
    public static final String DECODER_OUTPUT_VAR_NAME = "decoderOutputNormalized";

    public TransformerDecoderGraphGenerator(String id, int sequenceLength, int hiddenSize, int intermediateLayerSize,
                                            int decoderLayersAmount, int attentionHeadsAmount, boolean fixDecoderWeights,
                                            double dropoutRate) {
        super(sequenceLength, hiddenSize, intermediateLayerSize, decoderLayersAmount, attentionHeadsAmount, fixDecoderWeights, dropoutRate,
                id);
    }

    public SDVariable generateGraph(@Nonnull SameDiff sd, @Nonnull SDVariable batchInputTokenEmbeddings,
                                    @Nonnull SDVariable selfAttentionCausalMasks, @Nullable SDVariable positionalEmbeddings) {
        SDVariable lastLayerOutput = null;
        boolean firstLayerInitialized = false;

        try (var ignored = sd.withNameScope(id)) {
            var batchSize = getDimensionSize("batchSize", batchInputTokenEmbeddings, 0);
            var decoderSequenceLength = getDimensionSize("decoderSequenceLength", batchInputTokenEmbeddings, 1);
            var flatBatchSizeDecoder = batchSize.mul("flatBatchSize", decoderSequenceLength);
            var hiddenSizeSd = sd.constant("hiddenSize", hiddenSize);
            var attentionInputShape = sd.stack("attentionInputShape", 0, flatBatchSizeDecoder, hiddenSizeSd);
            var one = sd.constant("one", 1);
            var positionalEmbeddingsForAttention = ofNullable(positionalEmbeddings)
                    .map(embeddings -> sd.tile("positionalEmbeddingsForAttention", positionalEmbeddings, sd.stack(0, batchSize, one)))
                    .orElse(null);

            var selfAttentionLayerGraphGenerator = getSelfAttentionLayerGraphGenerator(sd, attentionHeadEmbeddingSize,
                    decoderSequenceLength, batchSize, positionalEmbeddingsForAttention, "default");
            var hiddenLayerBlockGraphGenerator = getHiddenLayerBlockGraphGenerator(sd);

            for (int i = 0; i < layersAmount; i++) {
                // Input for self-attention - either initial token embeddings, or the output of the last layer
                var layerInput = firstLayerInitialized ? lastLayerOutput : batchInputTokenEmbeddings.reshape(attentionInputShape);
                var selfAttentionResult = selfAttentionLayerGraphGenerator.generateGraph(i, layerInput, selfAttentionCausalMasks);
                lastLayerOutput = hiddenLayerBlockGraphGenerator.generateGraph(i, selfAttentionResult.selfAttentionResidualProduct())
                        .layerResidualProduct();
                firstLayerInitialized = true;
            }

            return normalizeLayerWithNoGain(sd, DECODER_OUTPUT_VAR_NAME, lastLayerOutput, 1, hiddenSize);
        }
    }

    public static class Builder extends TransformerBlockGraphGenerator.Builder<Builder, TransformerDecoderGraphGenerator> {
        public Builder(String id) {
            super(id);
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public TransformerDecoderGraphGenerator build() {
            return new TransformerDecoderGraphGenerator(id, sequenceLength, hiddenSize, intermediateLayerSize, layersAmount,
                    attentionHeadsAmount, fixWeights, dropoutRate);
        }
    }
}