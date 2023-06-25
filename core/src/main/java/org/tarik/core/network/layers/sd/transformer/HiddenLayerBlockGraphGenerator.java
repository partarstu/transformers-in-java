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

import java.util.List;

import static java.util.Optional.ofNullable;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;
import static org.tarik.utils.SdUtils.*;

/**
 * <p>
 * Classic simple, position-wise fully connected feed-forward network (sometimes called "Hidden" Layer) based on
 * the <a href="https://arxiv.org/pdf/1706.03762.pdf">original paper</a>.
 * </p>
 * <p>
 * This layer has 2 sub-layers. The first one is the extension of the attention output projections (dimension size) based
 * on {@link HiddenLayerBlockGraphGenerator#intermediateLayerSize}. The second one is the compression back to the original dimension size.
 * The layer normalization is applied to the input at the very beginning. Residual connections are applied based on the original input
 * before layer normalization.
 * </p>
 */
public class HiddenLayerBlockGraphGenerator {
    public static final String HIDDEN_INNER_LAYER_WEIGHTS = "hiddenInnerLayerWeights";
    public static final String HIDDEN_OUT_LAYER_WEIGHTS = "hiddenOutLayerWeights";
    public static final String HIDDEN_INNER_LAYER_BIAS = "hiddenInnerLayerBias";
    public static final String HIDDEN_OUT_LAYER_BIAS = "hiddenOutLayerBias";
    public static final String BLOCK_DIRECT_OUTPUT_VAR_NAME = "hiddenLayerFinalOutputNormalized";
    public static final String BLOCK_RESIDUAL_OUTPUT_VAR_NAME = "hiddenLayerResidualProductNormalized";

    private final SameDiff sameDiff;
    private final int hiddenSize;
    private final int intermediateLayerSize;
    private final boolean freezeAllWeights;
    private final double dropoutRate;
    private final String idSuffix;

    private HiddenLayerBlockGraphGenerator(SameDiff sameDiff, int hiddenSize, int intermediateLayerSize, boolean freezeAllWeights,
                                           double dropoutRate, String idSuffix) {
        this.sameDiff = sameDiff;
        this.hiddenSize = hiddenSize;
        this.intermediateLayerSize = intermediateLayerSize;
        this.freezeAllWeights = freezeAllWeights;
        this.dropoutRate = dropoutRate;
        this.idSuffix = idSuffix;
    }

    private String getSuffix(int layerIndex) {
        return "_" + layerIndex;
    }

    private String getUniqueName(String namePrefix) {
        return generateUniqueVariableName(sameDiff, namePrefix);
    }

    public HiddenLayerResult generateGraph(int layerIndex, SDVariable layerInput) {
        try (var ignored = sameDiff.withNameScope(String.format("hidden_ff_layer_%d%s", layerIndex,
                ofNullable(this.idSuffix).map(suf -> "_" + suf).orElse("")))) {
            var hiddenInnerLayerWeights = getOrCreateVariable(sameDiff, HIDDEN_INNER_LAYER_WEIGHTS + getSuffix(layerIndex),
                    name -> sameDiff.var(name, new XavierInitScheme('c', hiddenSize, intermediateLayerSize),
                            FLOAT, hiddenSize, intermediateLayerSize));
            var hiddenOutLayerWeights = getOrCreateVariable(sameDiff, HIDDEN_OUT_LAYER_WEIGHTS + getSuffix(layerIndex),
                    name -> sameDiff.var(name, new XavierInitScheme('c', intermediateLayerSize, hiddenSize),
                            FLOAT, intermediateLayerSize, hiddenSize));

            var hiddenInnerLayerBias = getOrCreateVariable(sameDiff, HIDDEN_INNER_LAYER_BIAS + getSuffix(layerIndex),
                    name -> sameDiff.zero(name, intermediateLayerSize).convertToVariable());
            sameDiff.updateVariableNameAndReference(hiddenInnerLayerBias, HIDDEN_INNER_LAYER_BIAS + getSuffix(layerIndex));
            var hiddenOutLayerBias = getOrCreateVariable(sameDiff, HIDDEN_OUT_LAYER_BIAS + getSuffix(layerIndex),
                    name -> sameDiff.zero(name, hiddenSize).convertToVariable());
            sameDiff.updateVariableNameAndReference(hiddenOutLayerBias, HIDDEN_OUT_LAYER_BIAS + getSuffix(layerIndex));

            if (freezeAllWeights) {
                sameDiff.convertToConstants(
                        List.of(hiddenInnerLayerWeights, hiddenOutLayerWeights, hiddenInnerLayerBias, hiddenOutLayerBias));
            }

            var layerInputNormalized = normalizeLayerWithNoGain(sameDiff,
                    getUniqueName("hiddenLayerInputNormalized" + getSuffix(layerIndex)), layerInput, 1, hiddenSize);
            var hiddenInnerLayerActivations = sameDiff.nn.linear(getUniqueName("hiddenInnerLayerActivations" + getSuffix(layerIndex)),
                    layerInputNormalized, hiddenInnerLayerWeights, hiddenInnerLayerBias);
            var hiddenInnerLayerOutput = sameDiff.nn.gelu(getUniqueName("hiddenInnerLayerOutput" + getSuffix(layerIndex)),
                    hiddenInnerLayerActivations);
            var hiddenFinalLayerOutput = sameDiff.nn.linear(getUniqueName("hiddenFinalLayerOutput" + getSuffix(layerIndex)),
                    hiddenInnerLayerOutput, hiddenOutLayerWeights, hiddenOutLayerBias);

            if (dropoutRate > 0) {
                hiddenFinalLayerOutput = sameDiff.nn().dropout(hiddenFinalLayerOutput, false, 1 - dropoutRate);
            }

            hiddenFinalLayerOutput.rename(getUniqueName(BLOCK_DIRECT_OUTPUT_VAR_NAME + getSuffix(layerIndex)));

            var layerResidualProduct = layerInput.add(getUniqueName(BLOCK_RESIDUAL_OUTPUT_VAR_NAME + getSuffix(layerIndex)),
                    hiddenFinalLayerOutput);

            return new HiddenLayerResult(hiddenFinalLayerOutput, layerResidualProduct);
        }
    }

    public record HiddenLayerResult(SDVariable layerOutput, SDVariable layerResidualProduct) {
    }

    public static final class Builder {
        private final SameDiff sameDiff;
        private int hiddenSize = 768;
        private int intermediateLayerSize = 1024;
        private double dropoutRate = 0;
        private boolean fixEncoderWeights;
        private String idSuffix;

        public Builder(SameDiff sameDiff) {
            this.sameDiff = sameDiff;
        }

        public Builder withHiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder withIntermediateLayerSize(int intermediateLayerSize) {
            this.intermediateLayerSize = intermediateLayerSize;
            return this;
        }

        public Builder withIdSuffix(String idSuffix) {
            this.idSuffix = idSuffix;
            return this;
        }

        public Builder withDropoutRate(double dropoutRate) {
            this.dropoutRate = dropoutRate;
            return this;
        }

        public Builder withFixedWeights(boolean fixEncoderWeights) {
            this.fixEncoderWeights = fixEncoderWeights;
            return this;
        }

        public HiddenLayerBlockGraphGenerator build() {
            return new HiddenLayerBlockGraphGenerator(sameDiff, hiddenSize, intermediateLayerSize, fixEncoderWeights, dropoutRate,
                    idSuffix);
        }
    }
}