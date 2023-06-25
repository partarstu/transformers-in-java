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

import static org.tarik.utils.SdUtils.*;

/**
 * A modified Decoder Block (Stack) adapted to be used in the Question-Answering generative models. It presumes the presence of the
 * encoder's output. The inputs to this model are separated for question and context (from which the answer should be derived)
 * Positional embeddings are optional. Traditional self-attention masks are used and are generated based on the specified simple token
 * presence masks for each sequence.
 */
public class TransformerQaDecoderGraphGenerator extends TransformerDecoderGraphGenerator {
    public static final String DECODER_OUTPUT_VAR_NAME = "qaDecoderOutputNormalized";

    public TransformerQaDecoderGraphGenerator(String id, int sequenceLength, int hiddenSize, int intermediateLayerSize,
                                              int decoderLayersAmount, int attentionHeadsAmount, boolean fixDecoderWeights,
                                              double dropoutRate) {
        super(id, sequenceLength, hiddenSize, intermediateLayerSize, decoderLayersAmount, attentionHeadsAmount, fixDecoderWeights,
                dropoutRate);
    }

    public SDVariable generateGraph(@Nonnull SameDiff sd, @Nonnull SDVariable decoderInputTokenEmbeddings,
                                    @Nonnull SDVariable decoderSelfAttentionMasks, @Nonnull SDVariable encoderQuestionTokensOutput,
                                    @Nonnull SDVariable encoderQuestionMasks, @Nonnull SDVariable encoderContextsTokensOutput,
                                    @Nonnull SDVariable encoderContextsMasks, SDVariable decoderPositionalEmbeddings,
                                    @Nonnull SDVariable causalMasksTemplate) {
        SDVariable lastLayerHiddenOutput = null;
        boolean firstLayerInitialized = false;

        try (var ignored = sd.withNameScope(id)) {
            var batchSize = getDimensionSize("batchSize", decoderInputTokenEmbeddings, 0);
            var decoderSequenceLength = getDimensionSize("decoderSequenceLength", decoderInputTokenEmbeddings, 1);
            var encoderQuestionTokensSequenceLength =
                    getDimensionSize("encoderQuestionTokensSequenceLength", encoderQuestionTokensOutput, 1);
            var encoderContextsTokensSequenceLength =
                    getDimensionSize("encoderContextsTokensSequenceLength", encoderContextsTokensOutput, 1);

            var flatBatchSizeDecoder = batchSize.mul("flatBatchSize", decoderSequenceLength);
            var flatBatchSizeEncoderQuestions = batchSize.mul("flatBatchSizeEncoderQuestions", encoderQuestionTokensSequenceLength);
            var flatBatchSizeEncoderContexts = batchSize.mul("flatBatchSizeEncoderContexts", encoderContextsTokensSequenceLength);
            var hiddenSizeSd = sd.constant("hiddenSize", hiddenSize);
            var attentionInputShape = sd.stack("attentionInputShape", 0, flatBatchSizeDecoder, hiddenSizeSd);
            var encoderQuestionsInputShape = sd.stack("encoderQuestionsInputShape", 0, flatBatchSizeEncoderQuestions, hiddenSizeSd);
            var encoderContextsInputShape = sd.stack("encoderContextsInputShape", 0, flatBatchSizeEncoderContexts, hiddenSizeSd);

            var one = getOrCreateConstant(sd, "one", 1);

            var decoderCausalSelfAttentionMasks = createSelfAttentionCausalMasks(sd, decoderSelfAttentionMasks, causalMasksTemplate);
            var ownPositionalEmbeddingsForAttention = decoderPositionalEmbeddings == null ? null :
                    getDecoderPositionalEmbeddings(sd, decoderPositionalEmbeddings, batchSize, decoderSequenceLength, one);
            var questionContextCrossAttentionMasks = createCrossAttentionMasks(sd, encoderQuestionMasks, encoderContextsMasks,
                    "qa1");
            var answerQuestionContextCrossAttentionMasks =
                    createCrossAttentionMasks(sd, decoderSelfAttentionMasks, encoderQuestionMasks, "qa2");

            var selfAttentionLayerGraphGenerator = getSelfAttentionLayerGraphGenerator(sd, attentionHeadEmbeddingSize,
                    decoderSequenceLength, batchSize, ownPositionalEmbeddingsForAttention, "default");
            var questionContextCrossAttentionGraphGenerator = getCrossAttentionLayerGraphGenerator(sd,
                    attentionHeadEmbeddingSize, encoderQuestionTokensSequenceLength, batchSize, null,
                    null, encoderContextsTokensSequenceLength, false, "question_context");
            var answerQuestionContextsCrossAttentionGraphGenerator = getCrossAttentionLayerGraphGenerator(sd,
                    attentionHeadEmbeddingSize, decoderSequenceLength, batchSize, null,
                    null, encoderQuestionTokensSequenceLength, false, "answer_question");
            var hiddenLayerBlockGraphGenerator = getHiddenLayerBlockGraphGenerator(sd);

            var encoderQuestionsOutputReshapedForAttention = encoderQuestionTokensOutput.reshape(encoderQuestionsInputShape)
                    .rename("encoderQuestionsOutputReshapedForAttention");
            var encoderContextsOutputReshapedForAttention = encoderContextsTokensOutput.reshape(encoderContextsInputShape)
                    .rename("encoderContextsOutputReshapedForAttention");
            var questionContextCrossAttentionResult = questionContextCrossAttentionGraphGenerator.generateGraph(0,
                    encoderQuestionsOutputReshapedForAttention, encoderContextsOutputReshapedForAttention,
                    questionContextCrossAttentionMasks);

            for (int i = 0; i < layersAmount; i++) {
                var layerInput =
                        firstLayerInitialized ? lastLayerHiddenOutput : decoderInputTokenEmbeddings.reshape(attentionInputShape);
                var selfAttentionResult = selfAttentionLayerGraphGenerator.generateGraph(i, layerInput, decoderCausalSelfAttentionMasks);
                var answerQuestionContextCrossAttentionResult = answerQuestionContextsCrossAttentionGraphGenerator.generateGraph(i,
                        selfAttentionResult.selfAttentionResidualProduct(),
                        questionContextCrossAttentionResult.attentionResult().attentionOutput(), answerQuestionContextCrossAttentionMasks);

                lastLayerHiddenOutput = hiddenLayerBlockGraphGenerator
                        .generateGraph(i, answerQuestionContextCrossAttentionResult.residualProduct()).layerResidualProduct();
                firstLayerInitialized = true;
            }

            return normalizeLayerWithNoGain(sd, DECODER_OUTPUT_VAR_NAME, lastLayerHiddenOutput, 1, hiddenSize);
        }
    }

    protected SDVariable getDecoderPositionalEmbeddings(SameDiff sd, SDVariable decoderPositionalEmbeddings, SDVariable batchSize,
                                                        SDVariable sequenceLengthSd, SDVariable one) {
        var zero = getOrCreateConstant(sd, "zero", 0);
        var inputPositions = getPositionsRange(sd, "input_positions", zero, sequenceLengthSd);
        var sequenceAdaptedPositionalEmbeddings =
                gather(sd, "sequenceDecoderPositionalEmbeddings", decoderPositionalEmbeddings, inputPositions, 0);
        return sd.tile("decoderPositionalEmbeddingsForAttention", sequenceAdaptedPositionalEmbeddings, sd.stack(0, batchSize, one));
    }

    public static final class Builder extends TransformerBlockGraphGenerator.Builder<Builder, TransformerQaDecoderGraphGenerator> {
        public Builder(String id) {
            super(id);
        }

        @Override
        protected Builder getInstance() {
            return this;
        }

        public TransformerQaDecoderGraphGenerator build() {
            return new TransformerQaDecoderGraphGenerator(id, sequenceLength, hiddenSize, intermediateLayerSize, layersAmount,
                    attentionHeadsAmount, fixWeights, dropoutRate);
        }
    }
}