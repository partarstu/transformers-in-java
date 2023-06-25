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
import org.nd4j.weightinit.impl.XavierInitScheme;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.LinkedList;
import java.util.List;

import static java.util.Optional.ofNullable;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;
import static org.tarik.utils.SdUtils.*;

/**
 * <p>
 * An experimental Encoder Block (Stack) as a modification of the classic one. This block has layers each of which consists of 2 sub-layers.
 * </p>
 * The first sub-layer is the "expert" layer. It calculates its own layer's "competence" scores based on the initial embeddings of
 * each token.
 * <p>
 * The second sub-layer is the classic Self-Attention layer. The original token embeddings are fed into this layer.
 * </p>
 * <p>
 * The scores produced by each "expert" layer are used together in the softmax function in to find out how "competent" each layer is in
 * regard to the token being processed. The resulting softmax scores are applied to the self-attention outputs of each layer in the same
 * manner as the self-attention scores are applied to each token's attention in order to produce the final token's hidden state. This
 * makes the mentioned above mechanism something like an "expert" attention.
 * </p>
 */
public class TransformerExpertBasedEncoderGraphGenerator extends TransformerEncoderGraphGenerator {
    public static final String COMPETENCE_SOFTMAX_SCORES_VAR_NAME = "encoderLayerCompetenceSoftmaxScores";
    public static final String ENCODER_OUTPUT_VAR_NAME = "expertBasedEncoderOutputNormalized";

    private String getSuffix(int layerIndex) {
        return "_" + layerIndex;
    }

    public TransformerExpertBasedEncoderGraphGenerator(String id, int sequenceLength, int hiddenSize, int intermediateLayerSize,
                                                       int encoderLayersAmount, int attentionHeadsAmount, boolean fixEncoderWeights,
                                                       double dropoutRate) {
        super(id, sequenceLength, hiddenSize, intermediateLayerSize, encoderLayersAmount,
                attentionHeadsAmount, fixEncoderWeights, dropoutRate);
    }

    public SDVariable generateGraph(@Nonnull SameDiff sd, @Nonnull SDVariable inputTokenMasks, @Nonnull SDVariable batchTokenEmbeddings,
                                    @Nullable SDVariable positionalEmbeddings) {
        List<LayerOutputScored> scoreLayerOutputs = new LinkedList<>();

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
            var layerInput = batchTokenEmbeddings.reshape(hiddenLayerInputShape);

            for (int i = 0; i < layersAmount; i++) {
                var layerCompetenceWeights = getOrCreateVariable(sd, "layerCompetenceWeights" + getSuffix(i),
                        name -> sd.var(name, new XavierInitScheme('c', hiddenSize, 1), FLOAT, hiddenSize, 1));
                var layerCompetenceBias = getOrCreateVariable(sd, "layerCompetenceBias" + getSuffix(i),
                        name -> sd.zero(name, 1).convertToVariable());
                var layerCompetenceScores = sd.nn.linear(generateUniqueVariableName(sd, "layerCompetenceScores" + getSuffix(i)),
                        layerInput, layerCompetenceWeights, layerCompetenceBias);
                var selfAttentionResult = selfAttentionLayerGraphGenerator.generateGraph(i, layerInput, attentionMasks);
                var selfAttentionResultActivations = sd.nn.gelu("selfAttentionActivations" + getSuffix(i),
                        selfAttentionResult.attentionResult().attentionOutput());

                scoreLayerOutputs.add(new LayerOutputScored(selfAttentionResultActivations, layerCompetenceScores));
            }

            // [layersAmount, batch_size*sequenceLength, hiddenSize]
            var encoderLayerOutputs = sd.stack("encoderLayerOutputs", 0,
                    scoreLayerOutputs.stream().map(LayerOutputScored::layerOutput).toArray(SDVariable[]::new));
            // [layersAmount, batch_size*sequenceLength, 1]
            var encoderLayerCompetenceAbsoluteScores = sd.stack("encoderLayerCompetenceAbsoluteScores", 0,
                    scoreLayerOutputs.stream().map(LayerOutputScored::competenceScores).toArray(SDVariable[]::new));
            // [layersAmount, batch_size*sequenceLength, 1]
            var encoderLayerCompetenceSoftmaxScores =
                    sd.nn().softmax(COMPETENCE_SOFTMAX_SCORES_VAR_NAME, encoderLayerCompetenceAbsoluteScores, 0);

            // [batch_size*sequenceLength, 1, layersAmount]
            var competenceSoftmaxScoresReshaped = sd.permute("competenceSoftmaxScoresReshaped",
                    encoderLayerCompetenceSoftmaxScores, 1, 2, 0);
            // [batch_size*sequenceLength, layersAmount, hiddenSize]
            var encoderLayerOutputsReshaped = sd.permute("encoderLayerOutputsReshaped", encoderLayerOutputs, 1, 0, 2);
            // [batch_size*sequenceLength, hiddenSize]
            var finalEmbeddingsBasedOnLayerCompetences = sd.squeeze("finalEmbeddingsBasedOnLayerCompetences",
                    competenceSoftmaxScoresReshaped.mmul(encoderLayerOutputsReshaped), 1);

            return normalizeLayerWithNoGain(sd, ENCODER_OUTPUT_VAR_NAME, finalEmbeddingsBasedOnLayerCompetences, 1, hiddenSize);
        }
    }

    private record LayerOutputScored(SDVariable layerOutput, SDVariable competenceScores) {
    }

    public static final class Builder extends TransformerBlockGraphGenerator.Builder<Builder, TransformerExpertBasedEncoderGraphGenerator> {

        public Builder(String id) {
            super(id);
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public TransformerExpertBasedEncoderGraphGenerator build() {
            return new TransformerExpertBasedEncoderGraphGenerator(id, sequenceLength,
                    hiddenSize, intermediateLayerSize, layersAmount, attentionHeadsAmount, fixWeights,
                    dropoutRate);
        }
    }
}