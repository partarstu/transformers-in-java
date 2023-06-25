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

package org.tarik.utils;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tarik.core.network.custom_ops.SimpleSoftMaxOp;

import java.util.Collection;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Function;

import static java.lang.String.format;
import static java.util.Optional.ofNullable;
import static org.nd4j.linalg.api.buffer.DataType.INT32;
import static org.nd4j.linalg.factory.Nd4j.createFromArray;
import static org.nd4j.linalg.factory.Nd4j.valueArrayOf;

/**
 * General utilities method collection which facilitate using Samediff
 */
public class SdUtils {

    public static SDVariable getLastDimensionMaxIndicesAsVector(SameDiff sd, SDVariable input) {
        var inputLastDimensionLength = sd.sizeAt(input, -1).castTo(INT32);
        var maxIndices = sd.argmax(input, -1);
        var indicesAsVectorShape = input.length().div(inputLastDimensionLength).castTo(INT32);
        var indicesAsVector = maxIndices.reshape(indicesAsVectorShape);
        var strides = getPositionsRange(sd, 0, indicesAsVector.length()).mul(inputLastDimensionLength);
        return indicesAsVector.add(strides);
    }

    public static SDVariable normalizeLayerWithNoGain(SameDiff sd, String resultName, SDVariable input, int dimension,
                                                      int... gainShape) {
        var gain = getGainConstant(sd, resultName + "_embedNormGain", 1, gainShape);
        return sd.nn().layerNorm(resultName, input, gain, false, dimension);
    }

    public static SDVariable getPositionsRange(SameDiff sd, String variableName, SDVariable from, SDVariable toExclusive) {
        SDVariable one = getOrCreateConstant(sd, "one", 1);
        return sd.range(variableName, from, toExclusive, one, INT32);
    }

    public static SDVariable getPositionsRange(SameDiff sd, SDVariable from, SDVariable toExclusive) {
        SDVariable one = getOrCreateConstant(sd, "one", 1);
        return sd.range(from, toExclusive, one, INT32);
    }

    public static SDVariable getPositionsRange(SameDiff sd, int from, SDVariable toExclusive) {
        return getPositionsRange(sd, sd.constant(from), toExclusive);
    }

    public static SDVariable getPositionsRange(SameDiff sd, String variableName, int from, int toExclusive) {
        return sd.range(variableName, from, toExclusive, 1, INT32);
    }

    public static SDVariable getPositionsRange(SameDiff sd, int from, int toExclusive) {
        return sd.range(from, toExclusive, 1, INT32);
    }

    public static SDVariable getReversePositionsRange(SameDiff sd, int from, SDVariable toExclusive) {
        return getPositionsRange(sd, sd.constant(from), toExclusive);
    }

    public static SDVariable getOrCreateConstant(SameDiff sd, String variableName, int value) {
        return getSdVariable(sd, variableName).map(SDVariable::convertToConstant)
                .orElseGet(() -> sd.updateVariableNameAndReference(sd.constant(value), variableName));
    }

    public static SDVariable getOrCreateConstant(SameDiff sd, String variableName, INDArray value) {
        return getSdVariable(sd, variableName).map(SDVariable::convertToConstant)
                .orElseGet(() -> sd.updateVariableNameAndReference(sd.constant(value), variableName));
    }

    public static SDVariable getOrCreateVariable(SameDiff sd, String variableName, Function<String, SDVariable> newValueProvider) {
        return getSdVariable(sd, variableName).orElseGet(() -> newValueProvider.apply(variableName));
    }

    public static SDVariable getDimensionSize(String resultName, SDVariable variable, int dimension) {
        return variable.shape().get(SDIndex.point(dimension)).castTo(resultName, DataType.INT32);
    }

    public static SDVariable getDimensionSize(SDVariable variable, int dimension) {
        return variable.shape().get(SDIndex.point(dimension)).castTo(DataType.INT32);
    }

    public static SDVariable gather(SameDiff sd, String name, SDVariable lookupTable, SDVariable indices, int dimension) {
        return sd.gather(name, lookupTable, indices, dimension);
    }

    public static SDVariable gather(SameDiff sd, SDVariable lookupTable, SDVariable indices, int dimension) {
        return sd.gather(lookupTable, indices, dimension);
    }

    public static SDVariable gather(SameDiff sd, String name, SDVariable lookupTable, int[] indices, int dimension) {
        return sd.gather(name, lookupTable, sd.constant(createFromArray(indices)), dimension);
    }

    public static SDVariable gather(SameDiff sd, SDVariable lookupTable, int[] indices, int dimension) {
        return sd.gather(lookupTable, sd.constant(createFromArray(indices)), dimension);
    }

    public static SDVariable gather(SameDiff sd, String name, SDVariable lookupTable, int index, int dimension) {
        return gather(sd, name, lookupTable, new int[]{index}, dimension);
    }

    public static SDVariable gather(SameDiff sd, SDVariable lookupTable, int index, int dimension) {
        return gather(sd, lookupTable, new int[]{index}, dimension);
    }

    public static SDVariable getGainConstant(SameDiff sd, String name, int gainValue, int... shape) {
        return getOrCreateConstant(sd, name, valueArrayOf(shape, gainValue));
    }

    public static SDVariable softmaxForLossCalculation(SameDiff sd, String name, SDVariable x, int dimension, SDVariable labels) {
        SDVariable out = new SimpleSoftMaxOp(sd, x, dimension, labels).outputVariable();
        return sd.updateVariableNameAndReference(out, name);
    }

    /**
     * Root Mean Square Layer Normalization according to
     * <a href="https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_tensorflow.py">original source</a>
     *
     * @param sd        SameDiff
     * @param layerSize amount of features across which the layer need to be normalized
     * @param input     input data
     * @param epsilon   eps. value
     * @return normalized tensor
     */
    public static SDVariable rmsNorm(SameDiff sd, String name, SDVariable input, int layerSize, double epsilon, int dimension) {
        var id = UUID.randomUUID().toString();
        try (var ignored = sd.withNameScope(id)) {
            var scale = sd.one("scale", layerSize).convertToVariable();
            var meanSquared = sd.mean(sd.math().square(input), true, dimension);
            return scale.mul(input).mul(name, sd.math().rsqrt(meanSquared.add(epsilon)));
        }
    }

    public static void fixAllVariables(Collection<SDVariable> variables) {
        variables.forEach(SDVariable::convertToConstant);
    }

    public static String generateUniqueVariableName(SameDiff sd, String variableNameBase) {
        var fullName = getVariableFullName(sd, variableNameBase);
        return sd.generateDistinctCustomVariableName(fullName);
    }

    public static Optional<SDVariable> getSdVariable(SameDiff sd, String variableName) {
        String fullName = getVariableFullName(sd, variableName);
        return Optional.of(fullName)
                .filter(sd::hasVariable)
                .map(sd::getVariable);
    }

    private static String getVariableFullName(SameDiff sd, String variableName) {
        return ofNullable(sd.currentNameScope())
                .map(nameScope -> format("%s/%s", nameScope, variableName))
                .orElse(variableName);
    }
}